function varargout = spmb_inv(varargin)
% Compute the inverse of batches of matrices
% 
% !!! very inefficient implementation
%
% FORMAT iA = spmb_inv(A)
%
% A  - (N x N) Input  batch of matrices
% Ai - (N x N) Output batch of matrices
%__________________________________________________________________________
%
% FORMAT spmb_inv(A,DIM)
% FORMAT spmb_inv(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (N x N x ...)
%           Ai  will have shape (N x N x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x N x N)
%           Ai  will have shape (... x N x N)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
%__________________________________________________________________________

% Yael Balbastre

% Parse "dim" keyword argument
if nargin >= 3 && isnumeric(args{3})
    dim     = args{3};
    args(3) = [];
else
    [dim,args] = spmb_parse_dim(args{:});
end

if dim > 0
    [varargout{1:nargout}] = left_inv(dim,args{:});
else
    [varargout{1:nargout}] = right_inv(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function iA = left_inv(d,A)

N = size(A,d);
if size(A,d+1) ~= N, error("Matrix must be square"); end

shape  = size(A);
lbatch = shape(1:d-1);
rbatch = shape(d+2:end);

iA = zeros(     [prod(lbatch) N N prod(rbatch)], class(A));
A  = reshape(A, [prod(lbatch) N N prod(rbatch)]);

for i=1:prod(lbatch)
for j=1:prod(rbatch)
    iA(i,:,:,j) = spm_unsqueeze(inv(spm_squeeze(A(i,:,:,j), [1 4])), [1 4]);
end
end

iA = reshape(iA, [lbatch N N rbatch]);

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function iA = right_inv(d,A)

N = size(A,ndims(A)+d);
if size(A,ndims(A)+d+1) ~= N, error("Matrix must be square"); end

shape = size(A);
lbatch = shape(1:ndims(A)+d-1);
rbatch = shape(ndims(A)+d+2:end);

iA = zeros(     [prod(lbatch) N N prod(rbatch)], class(A));
A  = reshape(A, [prod(lbatch) N N prod(rbatch)]);

for i=1:prod(lbatch)
for j=1:prod(rbatch)
    iA(i,:,:,j) = spm_unsqueeze(inv(spm_squeeze(A(i,:,:,j), [1 4])), [1 4]);
end
end

iA = reshape(iA, [lbatch N N rbatch]);

end