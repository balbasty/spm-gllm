classdef spm_volarray
% A virtual (stack of) volumes that may have different affines and shapes
% but share a world space.
%
% FORMAT A = spm_volarray(P,mat,dim)
%
% P    - A char or cell array of filenames
% mat  - (4 x 4) Affine matrix     [default: from first volume in P]
% dim  - (1 x 3) Volume dimensions [default: from first volume in P]
% hold - Interpolation method      [default: 1] (see spm_slice_vol) 
% dt   - Returned data type        [default: double]
%__________________________________________________________________________

% Yael Balbastre
% =========================================================================
properties
    vol
    mat
    dim
    hold
    dt
    msk
end
methods        
% =========================================================================
function obj = spm_volarray(P,mat,dim,hold,dt)
% FORMAT A = spm_volarray(P,mat,dim)
%
% P    - A char or cell array of filenames
% mat  - (4 x 4) Affine matrix     [default: from first volume in P]
% dim  - (1 x 3) Volume dimensions [default: from first volume in P]
% hold - Interpolation method      [default: 1] (see spm_slice_vol) 
% dt   - Returned data type        [default: double]
%__________________________________________________________________________
    obj.vol = spm_vol(P);
    obj.msk = [];
    if nargin > 1, obj.mat  = mat;
    else,          obj.mat  = obj.vol(1).mat;    end
    if nargin > 2, obj.dim  = dim;
    else,          obj.dim  = obj.vol(1).dim;    end
    if nargin > 3, obj.hold = hold;
    else,          obj.hold = 1;                 end
    if nargin > 4, obj.dt   = dt;
    else,          obj.dt   = 'double';          end
end
% =========================================================================
function dim = size(obj,d)
    dim = [obj.dim size(obj.vol)];
    msk = dim ~= 1;
    dim = dim(1:find(msk,1,'last'));
    switch length(dim)
        case 0,  dim = [1   1];
        case 1,  dim = [dim 1];
    end
    if nargin > 1
        if d > length(dim), dim = 1; 
        else,               dim = dim(d); end
    end
end
% =========================================================================
function nd = length(obj)
    nd = max(size(obj));
end
% =========================================================================
function nd = numel(obj)
    nd = prod(size(obj));
end
% =========================================================================
function nd = ndims(obj)
    nd = length(size(obj));
end
% =========================================================================
function dt = class(obj)
    dt = obj.dt;
end
% =========================================================================
function S = subsref(obj,subs)
    if subs(1).type == '.'
        F = obj.(subs(1).subs);
        if numel(subs) > 1
            S = subsref(F,subs(2:end));
        end
        return;
    end
    if ~all(subs.type == '()'), error('Nested indexing not supported'); end
    subs = subs.subs;
    if length(subs) < 4 && numel(obj.vol) > 1
        error('All first four dimensions must be indexed');
    elseif length(subs) < 3
        error('All first three dimensions must be indexed');
    end
    if any(cellfun(@(x) islogical(x) && ~isvector(x), subs(1:3)))
        error('Multidimensional logical indexing not supported');
    end

    % --------------
    % Select volumes
    if length(subs) < 3, subs = [subs {':'}]; end
    V = obj.vol(subs{4:end});

    % -------------
    % Select planes
    allz = 1:size(obj,3);
    if ischar(subs{3}),    subs{3} = allz;          end
    if islogical(subs{3}), subs{3} = allz(subs{3}); end

    % -----------
    % Select mask
    if ~isempty(obj.msk)
        msksubs = subs;
        for i=1:length(subs)
            if islogical(msksubs{i})
                if length(msksubs{i}) > size(obj.msk,1)
                    msksubs{i} = ':';
                end
            elseif isnumeric(msksubs{i})
                msksubs{i} = min(msksubs{i}, size(obj.msk,i));
            end
        end
        submsk = obj.msk(msksubs{:});
    end

    % -----------------
    % Loop over volumes
    S = cell(1,numel(V));
    for l=1:numel(V)
        v = V(l);

        % ----------------
        % Loop over planes
        W = cell(1,length(subs{3}));
        for k=1:length(subs{3})
            z      = subs{3}(k);
            Al     = obj.mat\v.mat\spm_matrix([0 0 z]);
            W{k}   = spm_slice_vol(v,Al,obj.dim(1:2),obj.hold);
            W{k}   = W{k}(subs{1:2});
        end
        S{l} = cat(3,W{:});

    end
    S     = cat(4,S{:});
    shape = [size(S) 1];
    shape = [shape(1:3) size(V)];
    S     = reshape(S, shape);

    % ----------
    % Apply mask
    if ~isempty(obj.msk)
        S(submsk) = NaN;
    end

end
% =========================================================================
function obj = subsasgn(obj,subs,dat)
    error('spm_varray are read only');
end
% =========================================================================
function A = numeric(obj)
    i = repmat({':'}, 1, min(4,ndims(obj)));
    A = obj(i{:});
end
% =========================================================================
function A = full(obj)
    A = numeric(obj);
end
% =========================================================================
function A = double(obj)
    A = double(numeric(obj));
end
% =========================================================================
function A = single(obj)
    A = single(numeric(obj));
end
end % methods
end % class