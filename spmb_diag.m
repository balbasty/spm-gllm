function varargout = spmb_diag(varargin)
% Extract the k-th diagonal of a batch of matrices
%
% FORMAT D = spmb_diag(A)
% FORMAT D = spmb_diag(A,k)
% FORMAT D = spmb_diag(A,k,i,j)
%
% A - (M x N)  Input batch of matrices
% k - (int)    Diagonal to extract             [0]
% i - (int)    Index of first matrix dimension [dim>0 ? dim   : dim-1]
% j - (int)    Index of other matrix dimension [dim>0 ? dim+1 : dim]
% D - (K x 1)  Output batch of diagonals
%__________________________________________________________________________
%
% FORMAT spmb_diag(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (M x N x ...)
%           D   will have shape     (K x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x M x N)
%           D   will have shape (... x K x 1)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
% 
%__________________________________________________________________________
%
% Diagonals are numbered as followed:
% * = 0: (j=i) main diagonal
% * > 0: (j>i) above the main diagonal 
% * > 0: (j<i) below the main diagonal
%
% Wide matrix:          Long matrix:            Square matrix:
%   [ 0  1  2  3          [ 0  1  2               [ 0  1  2  3
%    -1  0  1  2           -1  0  1                -1  0  1  2
%    -2 -1  0  1 ]         -2 -1  0                -2 -1  0  1
%                          -3 -2 -1 ]              -3 -2 -1  0 ]
%__________________________________________________________________________

% Yael Balbastre

% NOTE
% I have tried an alternative implementation that leveraged 
% batch_transpose followed by linear indexing, but found it slower 
% that this implementation, despite its use of a loop.

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_diag(dim,args{:});
else
    [varargout{1:nargout}] = right_diag(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function D = left_diag(d,A,k,i,j)

if nargin < 4
    i = d;
    j = d+1;
    if nargin < 3
        k = 0;
    end
elseif nargin == 4
    error("Either none or both matrix indices must be provided.")
end

% Accept negative "python-like" indices
% -------------------------------------
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

if i < d || j < d
    error("Matrix axes must be to the right of the batch dimensions.")
end

% Compute output shape
% --------------------
M = size(A,i);
N = size(A,j);
if k < 0
    L = M + k;
else
    L = N - k;
end
L = min(L,min(M,N));

Ashape = size(A);
lbatch = Ashape(1:d-1);
rbatch = Ashape(d:end);
rbatch([i-(d-1) j-(d-1)]) = [];

% Select diagonal elements
% ------------------------
D      = zeros([lbatch L rbatch 1], class(A));
Aslice = repmat({':'}, 1, ndims(A));
Dslice = repmat({':'}, 1, ndims(D));
for l=1:L
    Dslice{length(lbatch)+1} = l;
    if k >= 0
        Aslice{i} = l;
        Aslice{j} = l+k;
    else
        Aslice{i} = l-k;
        Aslice{j} = l;
    end
    D(Dslice{:}) = reshape(A(Aslice{:}), [lbatch 1 rbatch 1]);
end

end


% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function D = right_diag(d,A,k,i,j)

if nargin < 4
    i = d-1;
    j = d;
    if nargin < 3
        k = 0;
    end
elseif nargin == 4
    error("Either none or both matrix indices must be provided.")
end

% Accept negative "python-like" indices
% -------------------------------------
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

if i > ndims(A) + 1 + d || j > ndims(A) + 1 + d
    error("Matrix axes must be to the left of the batch dimensions.")
end

% Compute output shape
% --------------------
M = size(A,i);
N = size(A,j);
if k < 0
    L = M + k;
else
    L = N - k;
end
L = min(L,min(M,N));

Ashape = size(A);
rbatch = Ashape(end+d+2:end);
lbatch = Ashape(1:end+d+1);
lbatch([i j]) = [];

% Select diagonal elements
% ------------------------
D = zeros([lbatch L rbatch 1],class(A));
Aslice = repmat({':'}, 1, ndims(A));
Dslice = repmat({':'}, 1, ndims(D));
for l=1:L
    Dslice{length(lbatch)+1} = l;
    if k >= 0
        Aslice{i} = l;
        Aslice{j} = l+k;
    else
        Aslice{i} = l-k;
        Aslice{j} = l;
    end
    D(Dslice{:}) = reshape(A(Aslice{:}), [lbatch 1 rbatch 1]);
end

end