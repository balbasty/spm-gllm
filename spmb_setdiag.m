function varargout = spmb_setdiag(varargin)
% Set the k-th diagonal of a batch of matrices
%
% FORMAT A = spmb_setdiag(A,D)
% FORMAT A = spmb_setdiag(A,D,k)
% FORMAT A = spmb_setdiag(A,D,k,i,j)
%
% A - (M x N)  Input batch of matrices
% D - (K x 1)  Input batch of diagonals
% k - (int)    Diagonal to extract             [0]
% i - (int)    Index of first matrix dimension [dim>0 ? dim   : dim-1]
% j - (int)    Index of other matrix dimension [dim>0 ? dim+1 : dim]
%__________________________________________________________________________
%
% FORMAT spmb_setdiag(...,'dim',DIM)
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

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_setdiag(dim,args{:});
else
    [varargout{1:nargout}] = right_setdiag(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function A = left_setdiag(d,A,D,k,i,j)

if nargin < 5
    i = d;
    j = d+1;
    if nargin < 4
        k = 0;
    end
elseif nargin == 5
    error("spmb_setdiag should take either 2, 3 or 5 arguments")
end

% Accept negative "python-like" indices
% -------------------------------------
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

if i < d || j < d
    error("Matrix axes must be to the right of the batch dimensions.")
end

% Broadcast
% ---------
Ashape = size(A);
Albatch = Ashape(1:d-1);
Arbatch = Ashape(d+2:end);

Dshape  = size(D);
Dlbatch = Dshape(1:d-1);
Drbatch = Dshape(d+1:end);

N = size(A,i);
M = size(A,j);
L = size(D,length(Dlbatch)+1);

Albatch = [ones(1, max(0, length(Dlbatch) - length(Albatch))) Albatch];
Dlbatch = [ones(1, max(0, length(Albatch) - length(Dlbatch))) Dlbatch];
Arbatch = [Arbatch ones(1, max(0, length(Drbatch) - length(Arbatch)))];
Drbatch = [Drbatch ones(1, max(0, length(Arbatch) - length(Drbatch)))];

A = reshape(A, [Albatch N M Arbatch]);
D = reshape(D, [Dlbatch L   Drbatch]);

% Create larger output array if needed
% ------------------------------------
if any(Albatch < Dlbatch) || any(Arbatch < Drbatch)
    Alrep = max(1, Dlbatch ./ Albatch);
    Arrep = max(1, Drbatch ./ Arbatch);
    A = repmat(A, [Alrep 1 1 Arrep]);
end
Albatch = max(Albatch,Dlbatch);

% Select diagonal elements
% ------------------------
Aslice = repmat({':'}, 1, ndims(A));
Dslice = repmat({':'}, 1, ndims(D));
for l=1:L
    Dslice{length(Albatch)+1} = l;
    if k >= 0
        Aslice{i} = l;
        Aslice{j} = l+k;
    else
        Aslice{i} = l-k;
        Aslice{j} = l;
    end
    A(Aslice{:}) = reshape(D(Dslice{:}), [Dlbatch 1 1 Drbatch]);
end

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function A = right_setdiag(d,A,D,k,i,j)

if nargin < 5
    i = ndims(A)-1;
    j = ndims(A);
    if nargin < 4
        k = 0;
    end
elseif nargin == 5
    error("spmb_setdiag should take either 2, 3 or 5 arguments")
end

isvec = isvector(D);
if iscolumn(D), D = reshape(D,size(D,2),size(D,1)); end

% Accept negative "python-like" indices
% -------------------------------------
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

if i > ndims(A) + 1 + d || j > ndims(A) + 1 + d
    error("Matrix axes must be to the left of the batch dimensions.")
end

% Broadcast
% ---------
Ashape = size(A);
Albatch = Ashape(1:ndims(A)+d-1);
Arbatch = Ashape(ndims(A)+d+2:end);

Dshape  = size(D);
Dlbatch = Dshape(1:ndims(D)+d);
Drbatch = Dshape(ndims(D)+d+2:end);

N = size(A,i);
M = size(A,j);
L = size(D,length(Dlbatch)+1);

P = max(0, length(Dlbatch) - length(Albatch));
i = i + P;
j = j + P;

Albatch = [ones(1, max(0, length(Dlbatch) - length(Albatch))) Albatch];
Dlbatch = [ones(1, max(0, length(Albatch) - length(Dlbatch))) Dlbatch];
Arbatch = [Arbatch ones(1, max(0, length(Drbatch) - length(Arbatch)))];
Drbatch = [Drbatch ones(1, max(0, length(Arbatch) - length(Drbatch)))];

A = reshape(A, [Albatch N M Arbatch]);
D = reshape(D, [Dlbatch L   Drbatch]);

% Create larger output array if needed
% ------------------------------------
if any(Albatch < Dlbatch) || any(Arbatch < Drbatch)
    Alrep = max(1, Dlbatch ./ Albatch);
    Arrep = max(1, Drbatch ./ Arbatch);
    A = repmat(A, [Alrep 1 1 Arrep]);
end
Albatch = max(Albatch,Dlbatch);

% Select diagonal elements
% ------------------------
Aslice = repmat({':'}, 1, ndims(A));
Dslice = repmat({':'}, 1, ndims(D));
for l=1:L
    Dslice{length(Albatch)+1} = l;
    if k >= 0
        Aslice{i} = l;
        Aslice{j} = l+k;
    else
        Aslice{i} = l-k;
        Aslice{j} = l;
    end
    A(Aslice{:}) = reshape(D(Dslice{:}), [Dlbatch 1 1 Drbatch]);
end

if isvec && P > 0
    A = spm_squeeze(A,1);
end

end