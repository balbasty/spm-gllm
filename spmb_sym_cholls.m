function varargout = spmb_sym_cholls(varargin)
% Solve a symmetric linear system with a precomputed Cholesky decomposition
%
% FORMAT x = spmb_sym_cholls(R,y)
% 
% R - (N*(N+1)/2 x 1) Compact triangular matrix (from spmb_sym_chol)
% y - (N x 1)         Input vector
% x - (N x 1)         Output vector
%__________________________________________________________________________
%
% FORMAT spmb_sym_cholls(R,y,DIM)
% FORMAT spmb_sym_cholls(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           R should have shape   (N2 x ...)
%           y should have shape    (N x ...)
%           x   will have shape    (N x ...)
%
%      * If `-1`:
%           R should have shape (... x N2)
%           y should have shape (... x N)
%           x   will have shape (... x N)
%__________________________________________________________________________
%
% A triangular matrix stored in flattened form contains the diagonal first,
% followed by each column of the triangle
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ 0 b f ]
%                                       [ 0 0 c ]
%__________________________________________________________________________

% Yael Balbastre

% Parse "dim" keyword argument
args = varargin;
if nargin >= 3 && isnumeric(args{3})
    dim     = args{3};
    args(3) = [];
else
    [dim,args] = spmb_parse_dim(args{:});
end

% Cholesky decomposition
if dim > 0
    [varargout{1:nargout}] = left_cholls(dim,args{:});
else
    [varargout{1:nargout}] = right_cholls(dim,args{:});
end

end

% =========================================================================
function x = left_cholls(d,R,y)

asrow = isrow(y) && isvector(R);
if isrow(R), R = reshape(R, size(R,2), size(R,1)); end
if isrow(y), y = reshape(y, size(y,2), size(y,1)); end

[Rlbatch,N2,Rrbatch] = spmb_splitsize(R,d,1);
[ylbatch,N ,yrbatch] = spmb_splitsize(y,d,1);
[Rlbatch,ylbatch]    = spmb_pad_shapes(Rlbatch,ylbatch,'left');
[Rrbatch,yrbatch]    = spmb_pad_shapes(Rrbatch,yrbatch,'right');

R = reshape(R, [Rlbatch N2 Rrbatch]);
y = reshape(y, [ylbatch N  yrbatch]);

x = cholls(d,R,y);

if asrow, x = reshape(x, size(x,2), size(x,1)); end
end

% =========================================================================
function x = right_cholls(d,R,y)

ascol = iscolumn(y) && isvector(R);
if iscolumn(R), R = reshape(R, size(R,2), size(R,1)); end
if iscolumn(y), y = reshape(y, size(y,2), size(y,1)); end

[Rlbatch,N2,Rrbatch] = spmb_splitsize(R,d,1);
[ylbatch,N ,yrbatch] = spmb_splitsize(y,d,1);
[Rlbatch,ylbatch]    = spmb_pad_shapes(Rlbatch,ylbatch,'left');
[Rrbatch,yrbatch]    = spmb_pad_shapes(Rrbatch,yrbatch,'right');

R = reshape(R, [Rlbatch N2 Rrbatch]);
y = reshape(y, [ylbatch N  yrbatch]);

d = ndims(y)+d+1;
x = cholls(d,R,y);

if ascol, x = reshape(x, size(x,2), size(x,1)); end
end

% =========================================================================
function x = cholls(d,R,y)

N   = size(y,d);
idx = mapidx(N);
l   = repmat({':'}, 1, d-1);
r   = repmat({':'}, 1, ndims(R)-d);

Rshape  = size(R);
Rlbatch = Rshape(1:d-1);
Rrbatch = Rshape(d+1:end);

yshape  = size(R);
ylbatch = yshape(1:d-1);
yrbatch = yshape(d+1:end);

xlbatch = max(Rlbatch, ylbatch);
xrbatch = max(Rrbatch, yrbatch);

x = zeros([xlbatch N xrbatch], class(y));

for i=1:N
    ii = [l {i} r];
    sm = y(ii{:});
    for k=i-1:-1:1
        kk = [l {k} r];
        ik = [l {idx(i,k)} r];
        sm = sm - R(ik{:}) .* x(kk{:});
    end
    sm       = sm ./ R(ii{:});
    x(ii{:}) = sm;
end
for i=N:-1:1
    ii = [l {i} r];
    sm = x(ii{:});
    for k=i+1:N
        kk = [l {k} r];
        ki = [l {idx(k,i)} r];
        sm = sm - R(ki{:}) .* x(kk{:});
    end
    sm       = sm ./ R(ii{:});
    x(ii{:}) = sm;
end

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