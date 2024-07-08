function varargout = spmb_full2sym(varargin)
% Convert a full symmetric matrix to a sparse Hessian matrix
%
% FORMAT H = spmb_full2sym(F)
% 
% F - (K  x K)  Batch of full matrices
% H - (K2 x 1)  Batch of sparse matrices (K2 = (K*(K+1))/2)
%__________________________________________________________________________
%
% FORMAT spmb_full2sym(F,DIM)
% FORMAT spmb_full2sym(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           F should have shape   (K x K x ...)
%           H   will have shape      (K2 x ...)
%
%      * If `-1`:
%           F should have shape (... x K x K)
%           H   will have shape (... x K2)
%__________________________________________________________________________
%
% A symmetric matrix stored in flattened form contains the diagonal first,
% followed by each column of the lower triangle
%
%              [ a d e ]
%              [ d b f ]      =>    [ a b c d e f ]
%              [ e f c ]
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

if dim > 0
    [varargout{1:nargout}] = left_full2sym(dim,args{:});
else
    [varargout{1:nargout}] = right_full2sym(dim,args{:});
end

end

function idx = mapidx(K)
K2 = (K*(K+1))/2;
idx = zeros(K2, 1, 'uint64');
k = K+1;
for i=1:K
    idx(i) = i + (i-1)*K;
    for j=i+1:K 
        idx(k) = i + (j-1)*K;
        k = k + 1;
    end
end
end

function H = left_full2sym(d,F)
K           = size(F,d);
K2          = (K*(K+1))/2;
i           = mapidx(K);
shape       = size(F);
lbatch      = shape(1:d-1);
rbatch      = shape(d+2:end);
l           = repmat({':'}, 1, length(lbatch));
H           = zeros([lbatch K2 rbatch 1]);
F           = reshape(F, [lbatch K*K rbatch 1]);
H(l{:},:,:) = F(l{:},i,:);
end

function H = right_full2sym(d,F)
K           = size(F,ndims(F)+d);
K2          = (K*(K+1))/2;
i           = mapidx(K);
shape       = size(F);
lbatch      = shape(1:end+d-1);
rbatch      = shape(end+d+2:end);
l           = repmat({':'}, 1, length(lbatch));
H           = zeros([lbatch K2 rbatch 1]);
F           = reshape(F, [lbatch K*K rbatch 1]);
H(l{:},:,:) = F(l{:},i,:);
end