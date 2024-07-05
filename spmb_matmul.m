function varargout = spmb_matmul(varargin)
% Compute the matrix product between batches of matrices
%
% FORMAT C = spmb_matmul(A, B)
%
% A - (M x N) Input batch of left matrices
% B - (N x K) Input batch of right matrices
% C - (M x K) Output batch of matrices
%__________________________________________________________________________
%
% FORMAT spmb_matmul(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (M x N x ...)
%           B should have shape (N x K x ...)
%           C   will have shape (M x K x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x M x N)
%           B should have shape (... x N x K)
%           C   will have shape (... x M x K)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_matmul(dim,args{:});
else
    [varargout{1:nargout}] = right_matmul(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function C = left_matmul(d, A, B)

M = size(A,d);
N = size(A,d+1);
K = size(B,d+1);

Ashape = size(A);
Albatch = Ashape(1:d-1);
Arbatch = Ashape(d+2:end);

Bshape = size(B);
Blbatch = Bshape(1:d-1);
Brbatch = Bshape(d+2:end);

Albatch = [ones(1, max(0, length(Blbatch) - length(Albatch))) Albatch];
Blbatch = [ones(1, max(0, length(Albatch) - length(Blbatch))) Blbatch];
Arbatch = [Arbatch ones(1, max(0, length(Brbatch) - length(Arbatch)))];
Brbatch = [Brbatch ones(1, max(0, length(Arbatch) - length(Brbatch)))];
Clbatch = max(Albatch,Blbatch);
Crbatch = max(Arbatch,Brbatch);

A = reshape(A, [Albatch M N Arbatch]);
B = reshape(B, [Blbatch N K Brbatch]);
l = repmat({':'}, 1, length(Clbatch));
r = repmat({':'}, 1, length(Crbatch));

C = zeros([Clbatch M K Crbatch], class(A(1)*B(1)));
for n=1:N
    C = C + A(l{:}, :, n, r{:}) .* B(l{:}, n, :, r{:});
end

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function C = right_matmul(d, A, B)

M = size(A,ndims(A)+d);
N = size(A,ndims(A)+d+1);
K = size(B,ndims(B)+d+1);

Ashape = size(A);
Albatch = Ashape(1:ndims(A)+d-1);
Arbatch = Ashape(ndims(A)+d+2:end);

Bshape = size(B);
Blbatch = Bshape(1:ndims(B)+d-1);
Brbatch = Bshape(ndims(B)+d+2:end);

Albatch = [ones(1, max(0, length(Blbatch) - length(Albatch))) Albatch];
Blbatch = [ones(1, max(0, length(Albatch) - length(Blbatch))) Blbatch];
Arbatch = [Arbatch ones(1, max(0, length(Brbatch) - length(Arbatch)))];
Brbatch = [Brbatch ones(1, max(0, length(Arbatch) - length(Brbatch)))];
Clbatch = max(Albatch,Blbatch);
Crbatch = max(Arbatch,Brbatch);

A = reshape(A, [Albatch M N Arbatch]);
B = reshape(B, [Blbatch N K Brbatch]);
l = repmat({':'}, 1, length(Clbatch));
r = repmat({':'}, 1, length(Crbatch));

C = zeros([Clbatch M K Crbatch], class(A(1)*B(1)));
for n=1:N
    C = C + A(l{:}, :, n, r{:}) .* B(l{:}, n, :, r{:});
end

end