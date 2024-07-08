function y = spmb_matvec(A,x,varargin)
% Compute the matrix product between batches of matrices
%
% FORMAT y = spmb_matvec(A,x)
%
% A - (M x N) Input batch of matrices
% x - (N x 1) Input batch of vectors
% y - (M x 1) Output batch of vectors
%__________________________________________________________________________
%
% FORMAT spmb_matvec(A,x,DIM)
% FORMAT spmb_matvec(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (M x N x ...)
%           B should have shape (N x ...)
%           C   will have shape (M x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x M x N)
%           B should have shape (... x N)
%           C   will have shape (... x M)
%           Batch dimensions are implicitely padded on the left
%__________________________________________________________________________

% Yael Balbastre

args = varargin;
if ~isempty(args) >= 3 && isnumeric(args{1})
    dim     = args{1};
else
    [dim,~] = spmb_parse_dim(args{:});
end

if dim > 0, y = left_matvec(dim,A,x);
else,       y = right_matvec(dim,A,x); end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function y = left_matvec(d,A,x)

if isrow(x), x = reshape(x, size(x,2), size(x,1)); end

xshape  = size(x);
xlbatch = xshape(1:d-1);
xrbatch = xshape(d+1:end);
xl      = repmat({':'}, 1, length(xlbatch));
xr      = repmat({':'}, 1, length(xrbatch));

Ashape = size(A);
Albatch = Ashape(1:d-1);
Arbatch = Ashape(d+2:end);
Al      = repmat({':'}, 1, length(Albatch));
Ar      = repmat({':'}, 1, length(Arbatch));

Arbatch = [ones(1, max(0,length(xrbatch)-length(Arbatch))) Arbatch];
xrbatch = [ones(1, max(0,length(Arbatch)-length(xrbatch))) xrbatch];
yrbatch = max(Arbatch, xrbatch);
ylbatch = max(Albatch, xlbatch);

M = size(A, d);
N = size(A, d+1);

y = zeros([ylbatch M yrbatch 1], class(A(1)*x(1)));
for n=1:N
    A1 = reshape(A(Al{:},:,n,Ar{:}), [Albatch M Arbatch]);
    y = y + A1 .* x(xl{:},n,xr{:});
end

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function y = right_matvec(d,A,x)

if iscolumn(x), x = reshape(x, size(x,2), size(x,1)); end

xshape  = size(x);
xlbatch = xshape(1:ndims(x)+d);
xrbatch = xshape(ndims(x)+d+2:end);
xl      = repmat({':'}, 1, length(xlbatch));
xr      = repmat({':'}, 1, length(xrbatch));

Ashape = size(A);
Albatch = Ashape(1:ndims(A)+d-1);
Arbatch = Ashape(ndims(A)+d+2:end);
Al      = repmat({':'}, 1, length(Albatch));
Ar      = repmat({':'}, 1, length(Arbatch));

Albatch = [Albatch ones(1, max(0,length(xlbatch)-length(Albatch)))];
xlbatch = [xlbatch ones(1, max(0,length(Albatch)-length(xlbatch)))];
yrbatch = max(Arbatch, xrbatch);
ylbatch = max(Albatch, xlbatch);

M = size(A, ndims(A)+d);
N = size(A, ndims(A)+d+1);

A = reshape(A, [Albatch M N Arbatch]);
x = reshape(x, [xlbatch N   xrbatch]);
y = zeros([ylbatch M yrbatch 1], class(A(1)*x(1)));

for n=1:N
    A1 = reshape(A(Al{:},:,n,Ar{:}), [Albatch M Arbatch]);
    y = y + A1 .* x(xl{:},n,xr{:});
end

end