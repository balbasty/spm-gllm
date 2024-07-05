function varargout = spmb_sym_inner(varargin)
% Symmetric inner product x'*H*y, with H stored in a sparse form
%
% FORMAT A = spmb_sym_inner(x,H)
% FORMAT A = spmb_sym_inner(x,H,y)
% 
% x -  (K x 1)  Batch of input vectors
% H - (K2 x 1)  Batch of input sparse matrices
% y -  (K x 1)  Batch of input vectors  [x]
% t -  (1 x 1)  Batch of output inner products: t = x'*H*y
%__________________________________________________________________________
%
% K2 = (K*(K+1))/2
%__________________________________________________________________________
%
% FORMAT spmb_sym_inner(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           H should have shape      (K2 x ...)
%           x should have shape       (K x ...)
%           y should have shape       (K x ...)
%           t   will have shape       (1 x ...)
%
%      * If `-1`:
%           H should have shape (... x K2)
%           x should have shape (... x K)
%           y should have shape (... x K)
%           t   will have shape (... x 1)
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

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_inner(dim,args{:});
else
    [varargout{1:nargout}] = right_inner(dim,args{:});
end

end

% =========================================================================
function t = left_inner(d,x,H,y)
if nargin < 4, y = x; end
% asrow = isrow(x) || (~isvector(x) && isrow(H));
% if isrow(x), x = reshape(x, size(x,2), size(x,1)); end
% if isrow(y), y = reshape(y, size(y,2), size(y,1)); end
% if isrow(H), H = reshape(H, size(H,2), size(H,1)); end
K           = size(x,d);
K2          = (K*(K+1))/2;
if size(H,d) ~= K2 || size(y,d) ~= K
    msg = 'Inconsistant matrix sizes.';
    if isrow(x) || isrow(y) || isrow(H)
        msg = [msg ' ' 'Some inputs are row vectors that maybe should ' ...
                       'have been column vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:d-1);
Xrbatch     = Xshape(d+1:end);
Yshape      = size(y);
Ylbatch     = Yshape(1:d-1);
Yrbatch     = Yshape(d+1:end);
Hshape      = size(H);
Hlbatch     = Hshape(1:d-1);
Hrbatch     = Hshape(d+1:end);
nbatch      = max([length(Hrbatch) length(Xrbatch) length(Yrbatch)]);
Xrbatch     = [Xrbatch ones(1,nbatch-length(Xrbatch))];
Yrbatch     = [Yrbatch ones(1,nbatch-length(Yrbatch))];
Hrbatch     = [Hrbatch ones(1,nbatch-length(Hrbatch))];
Trbatch     = max(max(Xrbatch,Yrbatch),Hrbatch);
Tlbatch     = max(max(Xlbatch,Ylbatch),Hlbatch);
l           = repmat({':'}, 1, length(Tlbatch));
r           = repmat({':'}, 1, length(Trbatch));
t           = zeros([Tlbatch 1 Trbatch 1]);
k           = K+1;
for i=1:K
    t = t + x(l{:},i,r{:}) .* y(l{:},i,r{:}) .* H(l{:},i,r{:});
    for j=i+1:K
        t = t + x(l{:},i,r{:}) .* y(l{:},j,r{:}) .* H(l{:},k,r{:});
        t = t + x(l{:},j,r{:}) .* y(l{:},i,r{:}) .* H(l{:},k,r{:});
        k = k + 1;
    end
end
% if asrow, t = reshape(t, size(t,2), size(t,1)); end
end

% =========================================================================
function t = right_inner(d,x,H,y)
if nargin < 4, y = x; end
% ascol = iscolumn(x) || (~isvector(x) && iscolumn(H));
% if iscolumn(x), x = reshape(x, size(x,2), size(x,1)); end
% if iscolumn(y), y = reshape(y, size(y,2), size(y,1)); end
% if iscolumn(H), H = reshape(H, size(H,2), size(H,1)); end
K           = size(x,ndims(x)+d+1);
K2          = (K*(K+1))/2;
if size(H,ndims(x)+d+1) ~= K2 || size(y,ndims(x)+d+1) ~= K
    msg = 'Inconsistant matrix sizes.';
    if iscolumn(x) || iscolumn(y) || iscolumn(H)
        msg = [msg ' ' 'Some inputs are columns vectors that maybe ' ...
                       'should have been row vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:end+d);
Xrbatch     = Xshape(end+d+2:end);
Yshape      = size(y);
Ylbatch     = Yshape(1:end+d);
Yrbatch     = Yshape(end+d+2:end);
Hshape      = size(H);
Hlbatch     = Hshape(1:end+d);
Hrbatch     = Hshape(end+d+2:end);
nbatch      = max([length(Hlbatch) length(Xlbatch) length(Ylbatch)]);
Xlbatch     = [ones(1,nbatch-length(Xlbatch)) Xlbatch];
Ylbatch     = [ones(1,nbatch-length(Ylbatch)) Ylbatch];
Hlbatch     = [ones(1,nbatch-length(Hlbatch)) Hlbatch];
Trbatch     = max(max(Xrbatch,Yrbatch),Hrbatch);
Tlbatch     = max(max(Xlbatch,Ylbatch),Hlbatch);
x           = reshape(x, [Xlbatch K  Xrbatch 1]);
y           = reshape(y, [Ylbatch K  Yrbatch 1]);
H           = reshape(H, [Hlbatch K2 Hrbatch 1]);
l           = repmat({':'}, 1, length(Tlbatch));
r           = repmat({':'}, 1, length(Trbatch));
t           = zeros([Tlbatch 1 Trbatch 1]);
k           = K+1;
for i=1:K
    t = t + x(l{:},i,r{:}) .* y(l{:},i,r{:}) .* H(l{:},i,r{:});
    for j=i+1:K
        t = t + x(l{:},i,r{:}) .* y(l{:},j,r{:}) .* H(l{:},k,r{:});
        t = t + x(l{:},j,r{:}) .* y(l{:},i,r{:}) .* H(l{:},k,r{:});
        k = k + 1;
    end
end
% if ascol, t = reshape(t, size(t,2), size(t,1)); end
end