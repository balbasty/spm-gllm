function B = spm_unsqueeze(A,dims,nbs)
% Insert singleton dimensions
%
% FORMAT B = spm_unsqueeze(A,d)
% FORMAT B = spm_unsqueeze(A,d,n)
%
% A - (array)  Input array
% d - (vector) Dimensions to unsqueeze
% n - (vector) Number of dimensions to insert in each position [1]
% B - (array)  Unsqueezed array
%__________________________________________________________________________
%
% Negative "python-like" dimensions are accepted and index from the back.
% However, they are shifted by one, such that
%   * If a dimension is provided with _positive_ indexing, the singleton
%     is inserted _before_ the old dimension at this index.
%   * If a dimension is provided with _negative_ indexing, the singleton
%     is inserted _after_ the old dimension at this index.
%
% Hence
%   * -1 => ndims(A) + 1
%   * -2 => ndims(A)
%   * -3 => ndims(A) - 1
%   * etc.
%
% For example:
%   * unsqueeze(A: (N x M), 1)  -> (1 x N x M)
%   * unsqueeze(A: (N x M), 2)  -> (N x 1 x M)
%   * unsqueeze(A: (N x M), -1) -> (N x M x 1)
%   * unsqueeze(A: (N x M), -2) -> (N x 1 x M)
%__________________________________________________________________________

% Yael Balbastre

if nargin < 3, nbs = 1; end

% Accept negative "python-like" indices
dims(dims < 0) = ndims(A) + 2 + dims(dims < 0);
dims = sort(dims, 'descend');
if any(dims <= 0)
    error('Out-of-bounds index');
end

% Ensure as many n as d
nbs = [nbs(:)' repmat(nbs(end), 1, max(0, length(dims)-length(nbs)))];
nbs = nbs(1:length(dims));

% No need to keep dimensions that are larger than ndims
keep = dims <= ndims(A);
dims = dims(keep);
nbs  = nbs(keep);

% Compute output shape
shape = size(A);
for i=1:length(dims)
    d = dims(i);
    n = nbs(i);
    shape = [shape(1:d-1) ones(1,n) shape(d:end)];
end

% Reshape
B = reshape(A, shape);

end