%% Clear workspace
clear all
close all
clc

%% Add paths
addpath /opt/fieldtrip_github/
ft_defaults
addpath /opt/ESIsoftware/matlab/tdt_preprocessing/
addpath /mnt/hpc/opt/ESIsoftware/matlab/esi-nbf
addpath /opt/ESIsoftware/matlab/slurmfun/
addpath /mnt/hpc/projects/MWzeronoise/Analysis/4Shivangi/software/syncopy-matlab-0.1b1
clc

%% Session folders
datafolder = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/neural_data/stimAalign_cut/full_length';
resultfolder = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/In_matlab';

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

sesspaths = cellfun(@(S) fullfile(datafolder, S), sesslist, 'Uniform', 0);
resultpaths = cellfun(@(S) fullfile(resultfolder, S), sesslist, 'Uniform', 0);

%% Loop over sessions to convert to fieldtrip structure

for isess = 1:length(sesslist)
    fprintf('Processing session %s...\n', sesslist{isess});
    
    % Go to analog data folder
    analog_folder = fullfile(sesspaths{isess}, 'datalfp.spy');
    cd(analog_folder)
    
    filename = 'datalfp';
    [loaded.data, loaded.trl, loaded.spyInfo, loaded.time] = ...
        load_spy_time([filename '.analog']);
    
    data = loaded.data;
    trl  = loaded.trl;
    time = loaded.time;        
    nChannels = size(data, 2);
    nTrials   = size(trl, 1);
    
    % Compute max trial length
    samplesPerTrial = max(trl(:,2) - trl(:,1) + 1);
    
    % Preallocate trial matrix
    data_trials = nan(samplesPerTrial, nChannels, nTrials, 'single');
    
    for i = 1:nTrials
        beg = trl(i,1);
        end_ = trl(i,2);
        len = end_ - beg + 1;
        
        % Sanity check: bounds
        if end_ <= size(data,1)
            data_trials(1:len,:,i) = data(beg:end_, :);
        end
    end
    
    % Convert to FieldTrip structure
    [samplesPerTrial, nChannels, nTrials] = size(data_trials);
    fsample = double(loaded.spyInfo.samplerate);
    
    data_ft = [];
    data_ft.trial = cell(1, nTrials);
    data_ft.time  = time';
    data_ft.sampleinfo = zeros(nTrials, 2);
    data_ft.label = arrayfun(@(c) sprintf('channel_%03d', c), 1:nChannels, 'UniformOutput', false);
    data_ft.fsample = fsample;
    
    for i = 1:nTrials
        beg = trl(i,1);
        end_ = trl(i,2);
        len = end_ - beg + 1;
        
        % Trial data [channels x samples]
        data_ft.trial{i} = squeeze(data_trials(1:len,:,i))';
        
        % Sample range
        data_ft.sampleinfo(i,:) = [beg, end_];
    end
    
    % Add trial info
    cd(sesspaths{isess})
    load('Trial_Info.mat');

    data_ft.cfg = [];
    data_ft.trialinfo = [trial_info.Trial_Number(:)-999, ...
                    trial_info.Reward(:), ...
                    trial_info.Difficulty(:)];
    
    % Save FieldTrip structure
    if ~exist(resultpaths{isess}, 'dir'), mkdir(resultpaths{isess}); end
    cd(resultpaths{isess});
    % ESIsave('data_ft.mat', 'data_ft');
    fprintf('Saved FieldTrip data for session %s\n', sesslist{isess});
end
