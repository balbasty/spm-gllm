function iA = spmb_sym_cholinv(R,varargin)
% Invert a batch of (compact) positive-definite matrices
% with a precomputed Cholesky decomposition
%
% FORMAT iA = spmb_sym_inv(R,y)
% 
% R  - (N*(N+1)/2 x 1) Input   compact triangular matrix (from spmb_sym_chol)
% iA - (N*(N+1)/2 x 1) Inverse compact matrix
%__________________________________________________________________________
%
% FORMAT spmb_sym_inv(R,DIM)
% FORMAT spmb_sym_inv(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape   (N2 x ...)
%           R   will have shape   (N2 x ...)
%
%      * If `-1`:
%           A should have shape (... x N2)
%           R   will have shape (... x N2)
%__________________________________________________________________________
%
% A symmetric matrix stored in flattened form contains the diagonal first,
% followed by each column of the lower triangle
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ d b f ]
%                                       [ e f c ]
%__________________________________________________________________________

% Yael Balbastre

% Parse "dim" keyword argument
args = varargin;
if nargin >= 2 && isnumeric(args{1})
    dim     = args{1};
else
    [dim,~] = spmb_parse_dim(args{:});
end

% Cholesky decomposition
if dim > 0
    iA = left_inv(dim,R);
else
    iA = right_inv(dim,R);
end

end

% =========================================================================
function iA = left_inv(d,R)
asrow = isrow(R);
if isrow(R), R  = reshape(R, size(R,2), size(R,1));    end
             iA = inv(d,R);
if asrow,    iA = reshape(iA, size(iA,2), size(iA,1)); end
end

% =========================================================================
function iA = right_inv(d,R)
ascol = iscolumn(R);
if iscolumn(R), R  = reshape(R, size(R,2), size(R,1));    end
                iA = inv(d,R);
if ascol,       iA = reshape(iA, size(iA,2), size(iA,1)); end
end

% =========================================================================
function iA = inv(d,R)

[lbatch,N2,rbatch] = spmb_splitsize(R,d,1);
if d < 0, d = ndims(R)+d+1; end

N   = findK(N2);
idx = mapidx(N);
l   = repmat({':'}, 1, d-1);
r   = repmat({':'}, 1, ndims(R)-d);

iA  = zeros([lbatch N2 rbatch 1], class(R));
x   = zeros([lbatch N  rbatch 1], class(R));

for m=1:N
    for i=1:N
        ii = [l {i} r];
        sm = (i == m);
        for k=i-1:-1:1
            kk = [l {k}        r];
            ik = [l {idx(i,k)} r];
            sm = sm - R(ik{:}) .* x(kk{:});
        end
        sm        = sm ./ R(ii{:});
        x(ii{:}) = sm;
    end
    for i=N:-1:m
        ii = [l {i} r];
        sm = x(ii{:});
        for k=i+1:N
            kk = [l {k}        r];
            ki = [l {idx(k,i)} r];
            sm = sm - R(ki{:}) .* x(kk{:});
        end
        sm        = sm ./ R(ii{:});
        im        = [l {idx(i,m)} r];
        x(ii{:})  = sm;
        iA(im{:}) = sm;
    end
end

end

% =========================================================================
function K = findK(K2)
K = (sqrt(1 + 8*K2) - 1)/2;
end

% =========================================================================
function idx = mapidx(K)
idx = zeros(K, 'uint64');
k = K+1;
for i=1:K
    idx(i,i) = i;
    for j=i+1:K 
        idx(i,j) = k;
        idx(j,i) = k;
        k = k + 1;
    end
end
end