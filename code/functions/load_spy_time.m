function [data, trl, spyInfo, time] = load_spy_time(inFile)
% LOAD_SPY Load data, trial definitions, and per-trial time vectors from HDF5/JSON Syncopy files
%
%   [data, trl, spyInfo, time] = load_spy(inFile)
%
% INPUT
% -----
%   inFile : optional, filename of INFO or HDF5 file
%            If not provided, a file selector will show up.
%
% OUTPUT
% ------
%   data    : data array
%   trl     : [nTrials x 3+N] trial definition array
%   spyInfo : spy.SyncopyInfo object with metadata
%   time    : cell array {nTrials x 1} of per-trial time vectors
%
% See also spy.SyncopyInfo

%% --- Handle input
if nargin == 0
    [infoFile, pathname] = uigetfile({...
        '*.*.info', 'Syncopy Data Files (*.*.info)';...
        '*', 'All files (*)'}, ...
        'Pick a data file');
    if infoFile == 0; return; end
    inFile = fullfile(pathname, infoFile);
else
    if isa(inFile, 'string')
        inFile = char(inFile);
    elseif ~isa(inFile, 'char')
        error('Input has to be a file-name, not %s', class(inFile));
    end
end

%% --- Parse filename & load metadata
[folder, filestem, ext] = fileparts(inFile);
filenameTokens = split([filestem, ext], '.');
assert(length(filenameTokens) >= 2 && length(filenameTokens) <= 3, ...
    'Invalid filename %s. Must be *.ext or *.ext.info', inFile)

dataclassToken = filenameTokens{2};
filestem = filenameTokens{1};

infoFile = fullfile(folder, [filestem, '.', dataclassToken, '.info']);
spyInfo = spy.SyncopyInfo(infoFile);

hdfFile = fullfile(folder, spyInfo.filename);
assert(strcmp(fullfile(folder, [filestem, '.' dataclassToken]), hdfFile), ...
    'Filename mismatch between INFO file and actual filename: %s vs %s', spyInfo.filename, [filestem '.' dataclassToken]);

%% --- Load DATA
h5toc = h5info(hdfFile);
dset_names = {h5toc.Datasets.Name};
msk = ~strcmp(dset_names, 'trialdefinition');
dclass = dset_names{msk};
ndim = length(h5toc.Datasets(msk).Dataspace.Size);

data = permute(h5read(hdfFile, ['/', dclass]), ndim:-1:1);

%% --- Load TRIALDEFINITION
trl = h5read(hdfFile, '/trialdefinition')';
trl(:,1) = trl(:,1) + 1;  % correct zero-based start index

%% --- Reconstruct TIME vectors
if isprop(spyInfo, 'samplerate')
    fs = spyInfo.samplerate;
else
    fs = h5readatt(hdfFile, '/', 'samplerate');
end

nTrials = size(trl,1);
time = cell(nTrials,1);
for it = 1:nTrials
    nsamp = trl(it,2) - trl(it,1) + 1;  % number of samples
    offs  = trl(it,3);                   % offset
    time{it} = double((0:nsamp-1) + offs) / double(fs); % <-- float time points
end

%% --- Extract and check container attributes
h5attrs = h5toc.Attributes;
attrs = struct();
for iAttr = 1:length(h5attrs)
    name = h5attrs(iAttr).Name;
    value = h5attrs(iAttr).Value;
    if iscell(value) && numel(value) == 1
        value = value{1};
    end
    if ~ischar(value)
        value = value';
    end    
    name = strrep(name, '_', 'x0x5F_');
    attrs.(name) = value;         
    assert(isequal(spyInfo.(name), value), ...
        'JSON/HDF5 mismatch for attribute %s', name)
end

end
