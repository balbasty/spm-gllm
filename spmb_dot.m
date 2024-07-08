function varargout = spmb_dot(varargin)
% Compute the dot product between batches of vectors
%
% FORMAT C = spmb_dot(A, B)
%
% A - (N x 1) Input batch of left vectors
% B - (N x 1) Input batch of right vectors
% C - (1 x 1) Output batch of scalars
%__________________________________________________________________________
%
% FORMAT spmb_dot(...,'squeeze')
%
% Squeeze the dimension along which the dot product is performed.
%__________________________________________________________________________
%
% FORMAT spmb_dot(A,B,DIM)
% FORMAT spmb_dot(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (N x ...)
%           B should have shape (N x ...)
%           C   will have shape (1 x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x N)
%           B should have shape (... x N)
%           C   will have shape (... x 1)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
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

% Parse "squeeze" keyword argument
squeeze = false;
for i=length(args):-1:1
    key = args{i};
    if ischar(key) && strcmpi(key, 'squeeze')
        squeeze = true;
        args(i) = [];
        break
    end
end

% Perform dot product
if dim > 0
    C = left_dot(dim,args{:});
else
    C = right_dot(dim,args{:});
    dim = ndims(C) + d + 1;
end

% Squeeze reduced dimension
if squeeze
    C = spm_squeeze(C,dim);
end

% Return
varargout{1} = C;

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function C = left_dot(d, A, B)

N = size(A,d);

Ashape = size(A);
Albatch = Ashape(1:d-1);
Arbatch = Ashape(d+1:end);

Bshape = size(B);
Blbatch = Bshape(1:d-1);
Brbatch = Bshape(d+1:end);

Albatch = [ones(1, max(0, length(Blbatch) - length(Albatch))) Albatch];
Blbatch = [ones(1, max(0, length(Albatch) - length(Blbatch))) Blbatch];
Arbatch = [Arbatch ones(1, max(0, length(Brbatch) - length(Arbatch)))];
Brbatch = [Brbatch ones(1, max(0, length(Arbatch) - length(Brbatch)))];
Clbatch = max(Albatch,Blbatch);
Crbatch = max(Arbatch,Brbatch);

A = reshape(A, [Albatch N Arbatch 1]);
B = reshape(B, [Blbatch N Brbatch 1]);
l = repmat({':'}, 1, length(Clbatch));
r = repmat({':'}, 1, length(Crbatch));

C = zeros([Clbatch 1 Crbatch 1], class(A(1)*B(1)));
for n=1:N
    C = C + A(l{:}, n, r{:}) .* B(l{:}, n,  r{:});
end

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function C = right_dot(d, A, B)

N = size(A,ndims(A)+d+1);

Ashape = size(A);
Albatch = Ashape(1:ndims(A)+d);
Arbatch = Ashape(ndims(A)+d+2:end);

Bshape = size(B);
Blbatch = Bshape(1:ndims(B)+d);
Brbatch = Bshape(ndims(B)+d+2:end);

Albatch = [ones(1, max(0, length(Blbatch) - length(Albatch))) Albatch];
Blbatch = [ones(1, max(0, length(Albatch) - length(Blbatch))) Blbatch];
Arbatch = [Arbatch ones(1, max(0, length(Brbatch) - length(Arbatch)))];
Brbatch = [Brbatch ones(1, max(0, length(Arbatch) - length(Brbatch)))];
Clbatch = max(Albatch,Blbatch);
Crbatch = max(Arbatch,Brbatch);

A = reshape(A, [Albatch N Arbatch 1]);
B = reshape(B, [Blbatch N Brbatch 1]);
l = repmat({':'}, 1, length(Clbatch));
r = repmat({':'}, 1, length(Crbatch));

C = zeros([Clbatch 1 Crbatch 1], class(A(1)*B(1)));
for n=1:N
    C = C + A(l{:}, n, r{:}) .* B(l{:}, n, r{:});
end

end