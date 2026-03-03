% Clean-up Script for LFP Trial Data

clear all
close all
clc

%% Setup paths

addpath /opt/fieldtrip_github/
ft_defaults
addpath /opt/ESIsoftware/matlab/tdt_preprocessing/
addpath /mnt/hpc/opt/ESIsoftware/matlab/esi-nbf
clc

%% Paths

eventfolder = '/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length';
datafolder = '/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/Datasets/In_matlab';
figfolder = '/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/plots/preprocessing/artifact summary/cosmos';

% Step control %still need to clean sess 1
stepA = 0;
stepB = 0;
stepC = 1;
stepD = 1;

%% Detect session folders

cd(datafolder)
temp = dir;
sesslist = {};
ii = 0;
for i = 1:length(temp)
    if temp(i).isdir && ~startsWith(temp(i).name, '.')
        if all(isstrprop(temp(i).name, 'digit'))
            ii = ii + 1;
            sesslist{ii,1} = temp(i).name;
        end
    end
end

% Generate session paths
sesspaths = cellfun(@(S) fullfile(datafolder, S), sesslist, 'Uniform', 0);
eventpaths = cellfun(@(S) fullfile(eventfolder, S), sesslist, 'Uniform', 0);

%% Removing bad channels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% List of good channels in csv

cd('/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/plots/RFs')
temp = dir;                     
RFsess = {};                  
ii = 0;                         

for i = 1:length(temp)
    if temp(i).isdir && ~startsWith(temp(i).name, '.')
        if all(isstrprop(temp(i).name, 'digit')) % Folder name only digits
            ii = ii + 1;
            RFsess{ii,1} = temp(i).name;
        end
    end
end

for i = 1:length(RFsess)
    sessFolder = fullfile(pwd, RFsess{i});
    matFile = fullfile(sessFolder, 'goodChan_log.mat');
    
    if isfile(matFile)
        S = load(matFile);
        fn = fieldnames(S);
        goodChan_log = S.(fn{1});  
        
        % Save as CSV
        csvFile = fullfile(sessFolder, 'goodChan_log.csv');
        writematrix(goodChan_log, csvFile);
        
        fprintf('Saved: %s\n', csvFile);
    else
        fprintf('Missing: %s\n', matFile);
    end
end

%% Only keep the good channels

% Path to RF folders
rfPath = '/mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/plots/RFs';
rfFolders = dir(rfPath);
rfFolders = rfFolders([rfFolders.isdir] & ~ismember({rfFolders.name}, {'.', '..'}));
rfNames = {rfFolders.name};
rfDates = str2double(rfNames);  

for isess = 1:length(sesspaths)
    % Load current session data
    sessFile = fullfile(sesspaths{isess}, 'data_ft.mat');
    ESIload(sessFile);
    [~, sessShort, ~] = fileparts(sesspaths{isess});
    sessDate = str2double(sessShort);  
    
    % Find closest RF session folder
    [~, idxClosest] = min(abs(rfDates - sessDate));
    closestRFFolder = fullfile(rfPath, rfNames{idxClosest});
    
    % Load goodChan_log
    rfMatFile = dir(fullfile(closestRFFolder, '*.mat'));
    if isempty(rfMatFile)
        warning('No .mat file found in RF folder: %s', closestRFFolder);
        continue;
    end
    rfData = load(fullfile(closestRFFolder, 'goodChan_log'));
    mask = rfData.goodChan_log == 0;  % bad channels
    
    % Apply mask to all trials
       lfptrials = data_ft;
    for t = 1:length(lfptrials.trial)
        lfptrials.trial{t}(mask, :) = NaN;
    end
    
    % Save
    saveFile = fullfile(sesspaths{isess}, 'lfptrials.mat');
    save(saveFile, 'lfptrials', '-v7.3');  % -v7.3 is safe for large data
    fprintf('Saved lfptrials for session: %s\n', sesspaths{isess});
end

%% Trial lengths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Histogram of trial lengths across all sessions

allTrialLengths = [];  % initialize

for isess = 1:length(sesspaths)
    % Load session data
    sessFile = fullfile(sesspaths{isess}, 'data_ft.mat');
    ESIload(sessFile);
    lfptrials = data_ft;
    
    % Compute trial lengths
    for t = 1:length(lfptrials.trial)
        trlTime = lfptrials.time{t};
        allTrialLengths(end+1) = trlTime(end) - trlTime(1); %#ok<AGROW>
    end
end

% Plot histogram
figure;
histogram(allTrialLengths, 'BinWidth', 0.05);
xlabel('Trial Length (s)');
ylabel('Number of Trials');
title('Histogram of Trial Lengths Across All Sessions');
xlim([0 20]);  % limit x-axis to 20 s
grid on;

%% Figure out what time length we want to do the cleaning on

a_vals = -0.3:0.05:-0.1;   % possible start times
b_vals = 0.3:0.1:1.5;      % possible end times

alpha = 12; % Weight: how much to value trial length relative to trial count

% Preload all sessions and trial start/end times
numSessions = length(sesspaths);
allSessions = cell(numSessions,1);
trialStarts  = cell(numSessions,1);
trialEnds    = cell(numSessions,1);

for isess = 1:numSessions
    ESIload(fullfile(sesspaths{isess}, 'data_ft.mat')); 
    lfptrials = data_ft;
    allSessions{isess} = lfptrials;
    trialStarts{isess} = cellfun(@(t) t(1), lfptrials.time);
    trialEnds{isess}   = cellfun(@(t) t(end), lfptrials.time);
end

% Initialize tracking variables for optimization
maxScore = -Inf;
bestA = NaN;
bestB = NaN;
bestLength = 0;
bestValid = 0;

% Loop over all (a,b) combinations
for ai = 1:length(a_vals)
    for bi = 1:length(b_vals)
        a = a_vals(ai);
        b = b_vals(bi);
        if b <= a
            continue; % skip invalid ranges
        end
        
        totalValid = 0;
        
        % Count valid trials across sessions
        for isess = 1:numSessions
            validTrials = (trialStarts{isess} <= a & trialEnds{isess} >= b);
            totalValid = totalValid + sum(validTrials);
        end
        
        trialLength = b - a;
        
        % Weighted score: balance trial count & trial length
        score = totalValid + (100*alpha * trialLength);
        
        % Update if this combination is better
        if score > maxScore
            maxScore = score;
            bestA = a;
            bestB = b;
            bestLength = trialLength;
            bestValid = totalValid;
        end
    end
end

fprintf('Global optimal a = %.2f, b = %.2f, trial length = %.2f s, valid trials = %d (score=%.2f)\n', ...
    bestA, bestB, bestLength, bestValid, maxScore);

% Apply optimal (a,b) to each session and create results table
results = table('Size',[numSessions,5], ...
    'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'Session','NumValidTrials','TotalTrials','TrialLength','DiffTrials'});

globalValidTrials = cell(numSessions,1);

for isess = 1:numSessions
    lfptrials = allSessions{isess};
    
    % Find valid trials for chosen window
    validIdx = (trialStarts{isess} <= bestA & trialEnds{isess} >= bestB);
    
    cfg = [];
    cfg.trials = find(validIdx);
    cfg.latency = [bestA bestB];
    globalValidTrials{isess} = ft_selectdata(cfg, lfptrials);
    
    % Store results in table
    results.Session(isess)       = string(sesslist{isess});
    results.NumValidTrials(isess) = sum(validIdx);
    results.TotalTrials(isess)    = numel(lfptrials.trial);
    results.TrialLength(isess)    = bestB - bestA;
    results.DiffTrials(isess)     = numel(lfptrials.trial) - sum(validIdx);
end

disp(results);

%% Check the lengths of apple trials

all_durations = []; % store across sessions

for isess = 1:length(sesspaths)
    ESIload(fullfile(sesspaths{isess}, 'data_ft.mat'));
    
    % Load event markers
    cd(fullfile(eventpaths{isess}));
    T = readtable('EventMarkers.csv');
    all_events = cell(height(T),1);
    for i = 1:height(T)
        all_events{i} = str2double(strsplit(T.EventMarkers{i}, ','));
    end
    
    % Trial lengths 
    trial_durations = [];   
    trial_indices = [];   
    
    for itrial = 1:length(all_events)
        if any(all_events{itrial} == 3063)
            % duration = last time point - first time point
            len_sec = data_ft.time{itrial}(end) - data_ft.time{itrial}(1);
            
            trial_durations(end+1) = len_sec;
            trial_indices(end+1) = itrial;
        end
    end
    
    % Append to global list
    all_durations = [all_durations, trial_durations];
end

% Histogram across all sessions
figure;
histogram(all_durations, 'BinWidth', 0.5); % adjust bin width if needed
xlabel('Trial duration (s)');
ylabel('Count');
xlim([0 10]); 
title('Distribution of trial durations with eventmarker 3063 across sessions');

%% Clean data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step A: 
% Load data from each session. Plot and save all trials per channel
% This gives a good overview of problematic channels and trials with artifacts)

if stepA
    for isess = 1:length(sesspaths)
        disp(['reviewing session: ', sesslist{isess}])
        
        try
            sess_fig_dir = fullfile(figfolder, sesslist{isess}, '1200ms');
            if ~isfolder(sess_fig_dir)
                mkdir(sess_fig_dir)
            end
            
            % Load data
            ESIload(fullfile(sesspaths{isess}, 'data_ft.mat'));
            lfptrials = data_ft;
            
            % Find which trials are long enough
            validTrials = [];
            for t = 1:numel(lfptrials.trial)
                trlTime = lfptrials.time{t};
                if trlTime(1) <= -0.2 && trlTime(end) >= 1.2
                    validTrials(end+1) = t; %#ok<AGROW>
                end
            end
            
            % Now restrict and crop
            cfg = [];
            cfg.trials  = validTrials;
            cfg.latency = [-0.2 1.2];
            lfpTrials   = ft_selectdata(cfg, lfptrials);
            
            close all
            
            num_chans = length(lfpTrials.label);
            chans_per_fig = 32;
            num_figs = ceil(num_chans / chans_per_fig);
            
            for fig_idx = 1:num_figs
                FigH = figure('Position', get(0, 'Screensize'));
                
                start_chan = (fig_idx - 1) * chans_per_fig + 1;
                end_chan = min(fig_idx * chans_per_fig, num_chans);
                chan_range = start_chan:end_chan;
                
                for i = 1:length(chan_range)
                    ichan = chan_range(i);
                    subplot(8, 4, i)
                    for itrial = 1:length(lfpTrials.sampleinfo)
                        plot(lfpTrials.time{1,1}, lfpTrials.trial{1, itrial}(ichan, :));
                        hold on
                    end
                    title(['Ch ', num2str(ichan)])
                    xlim([min(lfpTrials.time{1,1}), max(lfpTrials.time{1,1})])
                end
                
                % Save figure
                cd(sess_fig_dir)
                set(FigH, 'Units', 'Inches');
                pos = get(FigH, 'Position');
                set(FigH, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
                    'PaperSize', [pos(3), pos(4)])
                print(FigH, [sesslist{isess}, '_lfp_allChan_array', num2str(fig_idx)], '-dpdf', '-r0')
                close(FigH)
            end
            
            clear lfpTrials
        catch ME
            warning(['Error in session: ', sesslist{isess}, ' - ', ME.message])
        end
    end
end

%% Step B:
% Run semi-automated, visual artifact rejection using summaries like var,
% 1/var etc

if stepB
    
    %% Clean up LFP data
    
    for isess = 1
        
        disp(['reviewing session: ',sesslist{isess}])

        try
            %% LFP data
            cd(sesspaths{isess}),
            ESIload(fullfile(sesspaths{isess}, 'data_ft.mat'));
            
            cfg = [];
            cfg.method = 'summary';
            cfg.keepchannel = 'nan';
            cfg.keeptrial   = 'nan';
            lfpTrials_clean1 = ft_rejectvisual(cfg, data_ft);
            
            cd(sesspaths{isess}),
            ESIsave('lfpTrials_clean1','lfpTrials_clean1')
            
            clear lfpTrials_clean1 data_ft
            close all
            
        catch
            continue
        end
    end
    
    %% plot LFP summaries after artifact rejection
    
    for isess = 1
        disp(['reviewing session: ', sesslist{isess}])
        
        try
            sess_fig_dir = fullfile(figfolder, sesslist{isess}, '1200ms');
            if ~isfolder(sess_fig_dir)
                mkdir(sess_fig_dir)
            end
            
            % Load data
            ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean1.mat'));
            
            % Find which trials are long enough
            validTrials = [];
            for t = 1:numel(lfpTrials_clean1.trial)
                trlTime = lfpTrials_clean1.time{t};
                if trlTime(1) <= -0.2 && trlTime(end) >= 1.2
                    validTrials(end+1) = t; %#ok<AGROW>
                end
            end
            
            % Now restrict and crop
            cfg = [];
            cfg.trials  = validTrials;
            cfg.latency = [-0.2 1.2];
            lfpTrials   = ft_selectdata(cfg, lfpTrials_clean1);
            
            close all
            
            num_chans = length(lfpTrials.label);
            chans_per_fig = 32;
            num_figs = ceil(num_chans / chans_per_fig);
            
            for fig_idx = 1:num_figs
                FigH = figure('Position', get(0, 'Screensize'));
                
                start_chan = (fig_idx - 1) * chans_per_fig + 1;
                end_chan = min(fig_idx * chans_per_fig, num_chans);
                chan_range = start_chan:end_chan;
                
                for i = 1:length(chan_range)
                    ichan = chan_range(i);
                    subplot(8, 4, i)
                    for itrial = 1:length(lfpTrials.sampleinfo)
                        plot(lfpTrials.time{1,1}, lfpTrials.trial{1, itrial}(ichan, :));
                        hold on
                    end
                    title(['Ch ', num2str(ichan)])
                    xlim([min(lfpTrials.time{1,1}), max(lfpTrials.time{1,1})])
                end
                
                % Save figure
                cd(sess_fig_dir)
                set(FigH, 'Units', 'Inches');
                pos = get(FigH, 'Position');
                set(FigH, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
                    'PaperSize', [pos(3), pos(4)])
                print(FigH, [sesslist{isess}, '_lfp_cleaned1_array', num2str(fig_idx)], '-dpdf', '-r0')
                close(FigH)
            end
            
            clear lfpTrials
        catch ME
            warning(['Error in session: ', sesslist{isess}, ' - ', ME.message])
        end
    end
end

%% Specify channels and plot them to find the big artifact's trial numberr

if stepC
    for isess = 1
        disp(['reviewing session: ', sesslist{isess}])
        
        selectedChannels = [1,2, 12, 13, 14, 16, 17, 24, 32, 33, 34, 35, 36, 37, 38, 39, 40,...
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,...
            61, 62, 63, 64, 82, 83, 84, 86, 97, 99, 100, 101, 109, 112, 113, 116, 123, 126, 128,...
            129, 130, 131, 134, 135, 140, 144, 145, 146, 147, 148, 149, 150, 160, 178];
        chansPerFig = 6;
        
        sess_fig_dir = fullfile(figfolder, sesslist{isess}, '1200ms');
        if ~isfolder(sess_fig_dir)
            mkdir(sess_fig_dir)
        end
        
        ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean1.mat'));
        lfpTrials_clean = lfpTrials_clean1;
        
        % Find which trials are long enough
        validTrials = [];
        for t = 1:numel(lfpTrials_clean.trial)
            trlTime = lfpTrials_clean.time{t};
            if trlTime(1) <= -0.2 && trlTime(end) >= 1.2
                validTrials(end+1) = t; %#ok<AGROW>
            end
        end
        
        % Now restrict and crop
        cfg = [];
        cfg.trials  = validTrials;
        cfg.latency = [-0.2 1.2];
        lfpTrials   = ft_selectdata(cfg, lfpTrials_clean);
        
        nSelectedChans = length(selectedChannels);
        nFigs = ceil(nSelectedChans / chansPerFig);
        
        for fig_idx = 1:nFigs
            FigH = figure('Position', get(0,'Screensize'));
            
            start_idx = (fig_idx - 1) * chansPerFig + 1;
            end_idx = min(fig_idx * chansPerFig, nSelectedChans);
            chan_range = selectedChannels(start_idx:end_idx);
            
            for i = 1:length(chan_range)
                ichan = chan_range(i);  % define ichan here
                subplot(ceil(length(chan_range)/2), 2, i)
                hold on
                
                for itrial = 1:length(lfpTrials.sampleinfo)
                    y = double(lfpTrials.trial{1, itrial}(ichan, :));
                    x = double(lfpTrials.time{1,1});
                    if isempty(y) || all(isnan(y)) || isempty(x)
                        continue
                    end
                    
                    plot(x, y, 'Color', [0 0 1 0.1]); % semi-transparent blue
                    
                    % Write trial number at max point
                    [~, idxMax] = max(y);
                    if ~isempty(idxMax)
                        text(x(idxMax), y(idxMax), num2str(validTrials(itrial)), ...
                            'FontSize', 6, 'Color', 'r', 'HorizontalAlignment', 'center')
                    end
                    
                    % Write trial number at min point
                    [~, idxMin] = min(y);
                    if ~isempty(idxMin)
                        text(x(idxMin), y(idxMin), num2str(validTrials(itrial)), ...
                            'FontSize', 6, 'Color', 'b', 'HorizontalAlignment', 'center')
                    end
                end
                
                title(['Ch ', num2str(ichan)])
                xlim([min(x), max(x)])
            end
            
            % Save figure
            cd(sess_fig_dir)
            set(FigH, 'Units', 'Inches');
            pos = get(FigH, 'Position');
            set(FigH, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
                'PaperSize', [pos(3), pos(4)])
            print(FigH, [sesslist{isess}, '_lfp_cleaned1_selected_chan_fig', num2str(fig_idx)], '-dpdf', '-r0')
            close(FigH)
        end
    end
    
     %% Remove specific trials and channels
    
     for isess = 1
         
         ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean1.mat'));
         lfpTrials_clean = lfpTrials_clean1;
         
         badTrials   = [19, 43, 80, 77, 82, 54, 68, 67, 66, 63, 58, 32, 62,...
             66, 75, 49, 46, 29, 32, 35, 73, 90, 92, 91, 61, 41, 56, 14, 48, ...
             83, 6, 11, 9, 13];
         badChannels = [1, 13, 16, 17, 34, 49, 50, 97, 112, 113, 128, 129, 134,...
             144, 145, 160, 32, 33]; 
         badSamples  = [];
         
         badTrials = unique(sort(badTrials));
         badChannels = unique(sort(badChannels));

         nTrials = numel(lfpTrials_clean.trial);
         
         for t = 1:nTrials
             
            thisTrial = lfpTrials_clean.trial{t};
            nChan = size(thisTrial, 1);
            nSamp = size(thisTrial, 2);
            
            % 1) Clean bad channels in all trials
            chIdx = intersect(badChannels, 1:nChan);
            if ~isempty(chIdx)
                if isempty(badSamples)
                    thisTrial(chIdx, :) = NaN;          % all samples
                else
                    sampIdx = badSamples(badSamples >= 1 & badSamples <= nSamp);
                    thisTrial(chIdx, sampIdx) = NaN;    % only specific samples
                end
            end
            
            % 2) Clean bad trials in all channels
            if ismember(t, badTrials)
                thisTrial(:,:) = NaN;
            end
            
            % assign back
            lfpTrials_clean.trial{t} = thisTrial;
         end
         
         lfpTrials_clean2 = lfpTrials_clean;
         ESIsave(fullfile(sesspaths{isess}, 'lfpTrials_clean2.mat'), 'lfpTrials_clean2');
         
     end
    
    %% plot LFP summaries after artifact rejection
    
    for isess = 1
        disp(['reviewing session: ', sesslist{isess}])
        
        try
            sess_fig_dir = fullfile(figfolder, sesslist{isess}, '1200ms');
            if ~isfolder(sess_fig_dir)
                mkdir(sess_fig_dir)
            end
            
            % Load data
             ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean2.mat'));
             lfpTrials_clean = lfpTrials_clean2;

             % Find which trials are long enough
            validTrials = [];
            for t = 1:numel(lfpTrials_clean.trial)
                trlTime = lfpTrials_clean.time{t};
                if trlTime(1) <= -0.2 && trlTime(end) >= 1.2
                    validTrials(end+1) = t; %#ok<AGROW>
                end
            end
            
            % Now restrict and crop
            cfg = [];
            cfg.trials  = validTrials;
            cfg.latency = [-0.2 1.2];
            lfpTrials   = ft_selectdata(cfg, lfpTrials_clean);
            
            close all
            
            num_chans = length(lfpTrials.label);
            chans_per_fig = 32;
            num_figs = ceil(num_chans / chans_per_fig);
            
            for fig_idx = 1:num_figs
                FigH = figure('Position', get(0, 'Screensize'));
                
                start_chan = (fig_idx - 1) * chans_per_fig + 1;
                end_chan = min(fig_idx * chans_per_fig, num_chans);
                chan_range = start_chan:end_chan;
                
                for i = 1:length(chan_range)
                    ichan = chan_range(i);
                    subplot(8, 4, i)
                    for itrial = 1:length(lfpTrials.sampleinfo)
                        plot(lfpTrials.time{1,1}, lfpTrials.trial{1, itrial}(ichan, :));
                        hold on
                    end
                    title(['Ch ', num2str(ichan)])
                    xlim([min(lfpTrials.time{1,1}), max(lfpTrials.time{1,1})])
                end
                
                % Save figure
                cd(sess_fig_dir)
                set(FigH, 'Units', 'Inches');
                pos = get(FigH, 'Position');
                set(FigH, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
                    'PaperSize', [pos(3), pos(4)])
                print(FigH, [sesslist{isess}, '_lfp_cleaned2_array', num2str(fig_idx)], '-dpdf', '-r0')
                close(FigH)
            end
            
            clear lfpTrials
        catch ME
            warning(['Error in session: ', sesslist{isess}, ' - ', ME.message])
        end
    end
end

%% Step D:
% Run semi-automated, visual artifact rejection using ft_rejectvisual to
% remove artfacts using trial or channel wise view

if stepD
    
    for isess = 1
        
        disp(['reviewing session: ',sesslist{isess}])
        
        try
            %% Clean data by going throughg channels or data one by one
            cd(sesspaths{isess}),
            ESIload(fullfile(sesspaths{isess},'lfpTrials_clean2'));
            
            cfg = [];
            cfg.method   = 'channel';
            cfg.keepchannel = 'nan';
            cfg.keeptrial   = 'nan';
            lfpTrials_clean4 = ft_rejectvisual(cfg,lfpTrials_clean2);
            
            cd(sesspaths{isess}),
            ESIsave('lfpTrials_clean3','lfpTrials_clean3')
            
            close all
            
        catch
            continue
        end
        
    end
    
    %% Remove specific trials and channels
    
    for isess = numel(sesspaths)
        
        ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean4.mat'));
        
        badTrials   = [417, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670];
        badChannels = [72, 67, 32, 163, 147, 131, 83,88, 11, 109, 140, 51, 150, 86, 68];
        badSamples  = [];
        
        badTrials = unique(sort(badTrials));
        badChannels = unique(sort(badChannels));
        nTrials = numel(lfpTrials_clean4.trial);
        
        for t = 1:nTrials
            
            thisTrial = lfpTrials_clean4.trial{t};
            nChan = size(thisTrial, 1);
            nSamp = size(thisTrial, 2);
            
            % 1) Clean bad channels in all trials
            chIdx = intersect(badChannels, 1:nChan);
            if ~isempty(chIdx)
                if isempty(badSamples)
                    thisTrial(chIdx, :) = NaN;          % all samples
                else
                    sampIdx = badSamples(badSamples >= 1 & badSamples <= nSamp);
                    thisTrial(chIdx, sampIdx) = NaN;    % only specific samples
                end
            end
            
            % 2) Clean bad trials in all channels
            if ismember(t, badTrials)
                thisTrial(:,:) = NaN;
            end
            
            % assign back
            lfpTrials_clean4.trial{t} = thisTrial;
        end
        
        lfpTrials_clean5 = lfpTrials_clean4;
        ESIsave(fullfile(sesspaths{isess}, 'lfpTrials_clean5.mat'), 'lfpTrials_clean5');
    end
    
    %% plot LFP summaries after artifact rejection
    
    for isess = length(sesspaths)
        disp(['reviewing session: ', sesslist{isess}])
        
        try
            sess_fig_dir = fullfile(figfolder, sesslist{isess}, '1200ms');
            if ~isfolder(sess_fig_dir)
                mkdir(sess_fig_dir)
            end
            
            % Load data
            ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean5.mat'));
            lfpTrials_clean = lfpTrials_clean5;
            
            % Find which trials are long enough
            validTrials = [];
            for t = 1:numel(lfpTrials_clean.trial)
                trlTime = lfpTrials_clean.time{t};
                if trlTime(1) <= -0.2 && trlTime(end) >= 1.2
                    validTrials(end+1) = t; %#ok<AGROW>
                end
            end
            
            % Now restrict and crop
            cfg = [];
            cfg.trials  = validTrials;
            cfg.latency = [-0.2 1.2];
            lfpTrials   = ft_selectdata(cfg, lfpTrials_clean);
            
            close all
            
            num_chans = length(lfpTrials.label);
            chans_per_fig = 32;
            num_figs = ceil(num_chans / chans_per_fig);
            
            for fig_idx = 1:num_figs
                FigH = figure('Position', get(0, 'Screensize'));
                
                start_chan = (fig_idx - 1) * chans_per_fig + 1;
                end_chan = min(fig_idx * chans_per_fig, num_chans);
                chan_range = start_chan:end_chan;
                
                for i = 1:length(chan_range)
                    ichan = chan_range(i);
                    subplot(8, 4, i)
                    for itrial = 1:length(lfpTrials.sampleinfo)
                        plot(lfpTrials.time{1,1}, lfpTrials.trial{1, itrial}(ichan, :));
                        hold on
                    end
                    title(['Ch ', num2str(ichan)])
                    xlim([min(lfpTrials.time{1,1}), max(lfpTrials.time{1,1})])
                end
                
                % Save figure
                cd(sess_fig_dir)
                set(FigH, 'Units', 'Inches');
                pos = get(FigH, 'Position');
                set(FigH, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
                    'PaperSize', [pos(3), pos(4)])
                print(FigH, [sesslist{isess}, '_lfp_cleaned5_array', num2str(fig_idx)], '-dpdf', '-r0')
                close(FigH)
            end
            
            clear lfpTrials
        catch ME
            warning(['Error in session: ', sesslist{isess}, ' - ', ME.message])
        end
    end
    
end

%% save without ESI function and make it syncopy competent

for isess = 5

    % Load cleaned LFP
    ESIload(fullfile(sesspaths{isess}, 'lfpTrials_clean3.mat'));
    cleaned_lfp = lfpTrials_clean3;

    % Just rename for compatibilitys
    data_ft = cleaned_lfp;

    % Save to Syncopy-ready MAT file
    out_path = fullfile(sesspaths{isess}, 'Data_FT.mat');
    cd(fullfile(sesspaths{isess}));
    save('tmp_ft.mat', 'data_ft', '-v7.3');
    ft_to_syncopy_mat('tmp_ft.mat', out_path);
    delete('tmp_ft.mat');   
end

