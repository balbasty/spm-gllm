function varargout = spmb_sym_chol(varargin)
% Cholesky decomposition of a batch of (compact) symmetric matrices
%
% FORMAT F = spmb_sym_choldc(A)
% 
% A - (N*(N+1)/2 x 1) Compact positive-definite matrix
% R - (N*(N+1)/2 x 1) Compact triangular matrix
%__________________________________________________________________________
%
% FORMAT spmb_sym_choldc(A,DIM)
% FORMAT spmb_sym_choldc(...,'dim',DIM)
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
%
% Similarly, an upper or lower triangular matrix can be stored in flattened
% form:
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ 0 b f ]
%                                       [ 0 0 c ]
%__________________________________________________________________________

% Yael Balbastre

% Parse "dim" keyword argument
args = varargin;
if nargin >= 2 && isnumeric(args{2})
    dim     = args{2};
    args(2) = [];
else
    [dim,args] = spmb_parse_dim(args{:});
end

% Cholesky decomposition
[varargout{1:nargout}] = chol(dim,args{:});

end

% =========================================================================
function R = chol(d,A)

asrow = false;
ascol = false;
if d < 0, ascol = iscolumn(A); else, asrow = isrow(A);   end
if asrow || ascol, A = reshape(A, size(A,2), size(A,1)); end

if d < 0, d = ndims(A)+d+1; end

R   = A;
N2  = size(A,d);
N   = findK(N2);
idx = mapidx(N);
l   = repmat({':'}, 1, d-1);
r   = repmat({':'}, 1, ndims(A)-d);

sm0 = 1e-7 * (sum(A(l{:},1:N,r{:}),d) + 1e-40);
sm0 = sm0 .* sm0;

for i=1:N

    sm = A(l{:},i,r{:});
    for k=i-1:-1:1
        ik = [l {idx(i,k)} r];
        sm = sm - R(ik{:}).^2;
    end
    sm(sm <= sm0) = sm0(sm <= sm0);
    ii = [l {i} r];
    R(ii{:}) = sqrt(sm);

    for j=i:N
        ij = [l {idx(i,j)} r];
        size(A);
        sm = A(ij{:});
        for k=i-1:-1:1
            ik = [l {idx(i,k)} r];
            jk = [l {idx(j,k)} r];
            sm = sm - R(ik{:}) .* R(jk{:});
        end
        R(ij{:}) = sm ./ R(ii{:});
    end
end

if asrow || ascol, R = reshape(R, size(R,2), size(R,1)); end
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