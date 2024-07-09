function y = spmb_sym_matvec(A,x,varargin)
% Matrix-vector multiplication, where the matrix is compact symmetric
%
% FORMAT y = spmb_sym_matvec(A,x)
% 
% A - (N2 x 1)  Batch of input compact symmetric matrices
% x -  (N x 1)  Batch of input vectors
% y -  (N x 1)  Batch of output vectors
%__________________________________________________________________________
%
% N2 = (N*(N+1))/2
%__________________________________________________________________________
%
% FORMAT spmb_sym_matvec(A,x,DIM)
% FORMAT spmb_sym_matvec(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (N2 x ...)
%           x should have shape (N x ...)
%           y   will have shape (N x ...)
%
%      * If `-1`:
%           A should have shape (... x N2)
%           X should have shape (... x N)
%           Y   will have shape (... x N)
%
%      * See `spmb_parse_dim` for more details.
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
if ~isempty(varargin) && isnumeric(varargin{1})
    dim     = varargin{1};
else
    [dim,~] = spmb_parse_dim(varargin{:});
end
if dim > 0
    y = left_matvec(dim,A,x);
else
    y = right_matvec(dim,A,x);
end
end

% =========================================================================
function y = left_matvec(d,A,x)
N           = size(x,d);
N2          = (N*(N+1))/2;
if size(A,d) ~= N2
    msg = 'Inconsistant matrix sizes.';
    if isrow(x) || isrow(A)
        msg = [msg ' ' 'Some inputs are row vectors that maybe should ' ...
                       'have been column vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:d-1);
Xrbatch     = Xshape(d+1:end);
Ashape      = size(A);
Albatch     = Ashape(1:d-1);
Arbatch     = Ashape(d+1:end);
nbatch      = max([length(Arbatch) length(Xrbatch)]);
Xrbatch     = [Xrbatch ones(1,nbatch-length(Xrbatch))];
Arbatch     = [Arbatch ones(1,nbatch-length(Arbatch))];
Yrbatch     = max(Xrbatch,Arbatch);
Ylbatch     = max(Xlbatch,Albatch);
y           = zeros([Ylbatch N Yrbatch 1]);
l           = repmat({':'}, 1, length(Ylbatch));
r           = repmat({':'}, 1, length(Yrbatch));
k           = N+1;
for i=1:N
    xi             = x(l{:},i,r{:});
    Ai             = A(l{:},i,r{:});
    y(l{:},i,r{:}) = y(l{:},i,r{:}) + xi .* Ai;
    for j=i+1:N
        xj             = x(l{:},j,r{:});
        Ak             = A(l{:},k,r{:});
        y(l{:},j,r{:}) = y(l{:},j,r{:}) + xi .* Ak;
        y(l{:},i,r{:}) = y(l{:},i,r{:}) + xj .* Ak;
        k = k + 1;
    end
end
end

% =========================================================================
function y = right_matvec(d,A,x)
N           = size(x,ndims(x)+d+1);
N2          = (N*(N+1))/2;
if size(A,ndims(A)+d+1) ~= N2
    msg = 'Inconsistant matrix sizes.';
    if iscolumn(x) || iscolumn(A)
        msg = [msg ' ' 'Some inputs are columns vectors that maybe ' ...
                       'should have been row vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:end+d);
Xrbatch     = Xshape(end+d+2:end);
Ashape      = size(A);
Albatch     = Ashape(1:end+d);
Arbatch     = Ashape(end+d+2:end);
nbatch      = max([length(Albatch) length(Xlbatch)]);
Xlbatch     = [ones(1,nbatch-length(Xlbatch)) Xlbatch];
Albatch     = [ones(1,nbatch-length(Albatch)) Albatch];
Yrbatch     = max(Xrbatch,Arbatch);
Ylbatch     = max(Xlbatch,Albatch);
x           = reshape(x, [Xlbatch N  Xrbatch 1]);
A           = reshape(A, [Albatch N2 Arbatch 1]);
y           = zeros([Ylbatch N Yrbatch 1]);
l           = repmat({':'}, 1, length(Ylbatch));
r           = repmat({':'}, 1, length(Yrbatch));
k           = N+1;
for i=1:N
    y(l{:},i,r{:}) = y(l{:},i,r{:}) + x(l{:},i,r{:}) .* A(l{:},i,r{:});
    for j=i+1:N
        y(l{:},j,r{:}) = y(l{:},j,r{:}) + x(l{:},i,r{:}) .* A(l{:},k,r{:});
        y(l{:},i,r{:}) = y(l{:},i,r{:}) + x(l{:},j,r{:}) .* A(l{:},k,r{:});
        k = k + 1;
    end
end
end