function ft_to_syncopy_mat(ft_mat_path, out_mat_path, varargin)
% Convert a FieldTrip data_ft.mat to a MAT-file readable by syncopy.load_ft_raw
% Uses original trial time vectors instead of generating new ones.

p = inputParser;
addRequired(p, 'ft_mat_path', @ischar);
addRequired(p, 'out_mat_path', @ischar);
addParameter(p, 'StructName', 'Data_FT', @ischar);
parse(p, ft_mat_path, out_mat_path, varargin{:});
opts = p.Results;

% Load input MAT file
tmp = load(ft_mat_path);
if ~isfield(tmp, 'data_ft')
    error('Input file does not contain variable ''data_ft''.');
end
data_ft = tmp.data_ft;

nTrials = numel(data_ft.trial);
firstTrial = data_ft.trial{1};
[nChannels, ~] = size(firstTrial);

% Build Syncopy-ready structure
S = struct();
S.trial = cell(1, nTrials);
for t = 1:nTrials
    tr = data_ft.trial{t};
    if size(tr,1) == nChannels
        S.trial{t} = double(tr);
    elseif size(tr,2) == nChannels
        S.trial{t} = double(tr');
    else
        error('Trial %d has mismatching dims [%d x %d]', t, size(tr,1), size(tr,2));
    end
end

% fsample
if isfield(data_ft, 'fsample') && ~isempty(data_ft.fsample)
    S.fsample = double(data_ft.fsample);
elseif isfield(data_ft, 'time') && iscell(data_ft.time) && ~isempty(data_ft.time{1})
    S.fsample = 1 / mean(diff(data_ft.time{1}));
else
    error('No fsample found and cannot infer from data_ft.time.');
end

% time: use original time vectors if present
S.time = cell(1, nTrials);
if isfield(data_ft, 'time') && iscell(data_ft.time)
    for t = 1:nTrials
        S.time{t} = double(data_ft.time{t}(:)');  % ensure row vector
    end
else
    % fallback (should rarely happen)
    for t = 1:nTrials
        nS = size(S.trial{t},2);
        S.time{t} = (0:(nS-1)) / S.fsample;
    end
end

% label
if isfield(data_ft, 'label')
    if iscell(data_ft.label) || isstring(data_ft.label)
        labels = cellstr(data_ft.label(:));
        if numel(labels) ~= nChannels
            labels = arrayfun(@(c) sprintf('channel_%03d', c), 1:nChannels, 'UniformOutput', false);
        end
    else
        labels = arrayfun(@(c) sprintf('channel_%03d', c), 1:nChannels, 'UniformOutput', false);
    end
else
    labels = arrayfun(@(c) sprintf('channel_%03d', c), 1:nChannels, 'UniformOutput', false);
end
S.label = labels;

% Drop trialinfo entirely
if isfield(S,'trialinfo')
    S = rmfield(S,'trialinfo');
end

% Optional cfg
if isfield(data_ft,'cfg')
    S.cfg = data_ft.cfg;
end

% Save as v7.3 for large files
VarName = opts.StructName;
eval(sprintf('%s = S;', VarName));
save(out_mat_path, VarName, '-v7.3');

fprintf('Wrote %s containing structure ''%s''.\n', out_mat_path, VarName);
end

function S = rmfield_cond(S, fld)
if isfield(S, fld)
    S = rmfield(S, fld);
end
end
