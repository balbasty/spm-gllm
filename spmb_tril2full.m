function varargout = spmb_tril2full(varargin)
% Convert a compact lower triangular matrix to a full matrix
%
% FORMAT F = spmb_tril2full(H)
% 
% H - (K2 x 1)  Batch of sparse matrices (K2 = (K*(K+1))/2)
% F - (K  x K)  Batch of full matrices
%__________________________________________________________________________
%
% FORMAT spmb_tril2full(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           H should have shape      (K2 x ...)
%           F   will have shape   (K x K x ...)
%
%      * If `-1`:
%           H should have shape (... x K2)
%           F   will have shape (... x K x K)
%__________________________________________________________________________
%
% A triangule matrix stored in flattened form contains the diagonal first,
% followed by each column of the lower triangle
%
%                                       [ a 0 0 ]
%       [ a b c d e f ]       =>        [ d b 0 ]
%                                       [ e f c ]
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_triu2full(dim,args{:});
else
    [varargout{1:nargout}] = right_triu2full(dim,args{:});
end

end

function K = findK(K2)
K = (sqrt(1 + 8*K2) - 1)/2;
end

function idx = mapidx(K)
idx = zeros((K*(K+1))/2, 1);
k = K+1;
for i=1:K
    idx(i) = (i-1)*K+i;
    for j=i+1:K
        idx(k) = (i-1)*K+j;
        k = k+1;
    end
end
end

function  F = left_triu2full(d,H)
asrow       = isrow(H);
if asrow, H = reshape(H, size(H,2), size(H,1)); end
K2          = size(H,d);
K           = findK(K2);
i           = mapidx(K);
shape       = size(H);
lbatch      = shape(1:d-1);
rbatch      = shape(d+1:end);
l           = repmat({':'}, 1, length(lbatch));
F           = zeros([lbatch K*K rbatch 1]);
i           = [l {i ':'}];
l           = [l {':' ':'}];
F(i{:})     = H(l{:});
F           = reshape(F, [lbatch K K rbatch]);
if asrow, F = reshape(F, size(F,2), size(F,1)); end
end

function  F = right_triu2full(d,H)
ascol       = iscolumn(H);
if ascol, H = reshape(H, size(H,2), size(H,1)); end
K2          = size(H,ndims(H)+d+1);
K           = findK(K2);
i           = mapidx(K);
shape       = size(H);
lbatch      = shape(1:end+d);
rbatch      = shape(end+d+2:end);
l           = repmat({':'}, 1, length(lbatch));
F           = zeros([lbatch K*K rbatch]);
i           = [l {i ':'}];
l           = [l {':' ':'}];
F(i{:})     = H(l{:});
F           = reshape(F, [lbatch K K rbatch]);
if ascol, F = reshape(F, size(F,ndims(F)), size(F,ndims(F)-1)); end
end