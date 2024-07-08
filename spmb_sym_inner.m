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
    if length(args) == 2
        [varargout{1:nargout}] = left_inner_xx(dim,args{:});
    else
        [varargout{1:nargout}] = left_inner_xy(dim,args{:});
    end
else
    if length(args) == 2
        [varargout{1:nargout}] = right_inner_xx(dim,args{:});
    else
        [varargout{1:nargout}] = right_inner_xy(dim,args{:});
    end
end

end

% =========================================================================
function t = left_inner_xy(d,x,H,y)
if nargin < 4, y = x; end
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
t           = zeros([Tlbatch 1 Trbatch 1], class(x(1)*y(1)*H(1)));
k           = K+1;
for i=1:K
    xi = x(l{:},i,r{:});
    yi = y(l{:},i,r{:});
    hi = H(l{:},i,r{:});
    t  = t + (xi .* yi) .* hi;
    for j=i+1:K
        xj = x(l{:},j,r{:});
        yj = y(l{:},j,r{:});
        hk = H(l{:},k,r{:});
        t  = t + (xi .* yj + xj .* yi) .* hk;
        k  = k + 1;
    end
end
end

% =========================================================================
function t = left_inner_xx(d,x,H)
K           = size(x,d);
K2          = (K*(K+1))/2;
if size(H,d) ~= K2
    msg = 'Inconsistant matrix sizes.';
    if isrow(x) || isrow(H)
        msg = [msg ' ' 'Some inputs are row vectors that maybe should ' ...
                       'have been column vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:d-1);
Xrbatch     = Xshape(d+1:end);
Hshape      = size(H);
Hlbatch     = Hshape(1:d-1);
Hrbatch     = Hshape(d+1:end);
nbatch      = max([length(Hrbatch) length(Xrbatch)]);
Xrbatch     = [Xrbatch ones(1,nbatch-length(Xrbatch))];
Hrbatch     = [Hrbatch ones(1,nbatch-length(Hrbatch))];
Trbatch     = max(Xrbatch,Hrbatch);
Tlbatch     = max(Xlbatch,Hlbatch);
l           = repmat({':'}, 1, length(Tlbatch));
r           = repmat({':'}, 1, length(Trbatch));
t           = zeros([Tlbatch 1 Trbatch 1], class(x(1)*H(1)));
k           = K+1;
for i=1:K
    xi = x(l{:},i,r{:});
    hi = H(l{:},i,r{:});
    t = t + (xi.^2) .* hi;
    for j=i+1:K
        xj = x(l{:},j,r{:});
        hk = H(l{:},k,r{:});
        t  = t + 2 * (xi .* xj) .* hk;
        k  = k + 1;
    end
end
end

% =========================================================================
function t = right_inner_xy(d,x,H,y)
if nargin < 4, y = x; end
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
t           = zeros([Tlbatch 1 Trbatch 1], class(x(1)*y(1)*H(1)));
k           = K+1;
for i=1:K
    xi = x(l{:},i,r{:});
    hi = H(l{:},i,r{:});
    t = t + (xi.^2) .* hi;
    for j=i+1:K
        xj = x(l{:},j,r{:});
        hk = H(l{:},k,r{:});
        t  = t + 2 * (xi .* xj) .* hk;
        k  = k + 1;
    end
end
end

% =========================================================================
function t = right_inner_xx(d,x,H)
K           = size(x,ndims(x)+d+1);
K2          = (K*(K+1))/2;
if size(H,ndims(x)+d+1) ~= K2
    msg = 'Inconsistant matrix sizes.';
    if iscolumn(x) || iscolumn(H)
        msg = [msg ' ' 'Some inputs are columns vectors that maybe ' ...
                       'should have been row vectors.'];
    end
    error(msg);
end
Xshape      = size(x);
Xlbatch     = Xshape(1:end+d);
Xrbatch     = Xshape(end+d+2:end);
Hshape      = size(H);
Hlbatch     = Hshape(1:end+d);
Hrbatch     = Hshape(end+d+2:end);
nbatch      = max([length(Hlbatch) length(Xlbatch)]);
Xlbatch     = [ones(1,nbatch-length(Xlbatch)) Xlbatch];
Hlbatch     = [ones(1,nbatch-length(Hlbatch)) Hlbatch];
Trbatch     = max(Xrbatch,Hrbatch);
Tlbatch     = max(Xlbatch,Hlbatch);
x           = reshape(x, [Xlbatch K  Xrbatch 1]);
H           = reshape(H, [Hlbatch K2 Hrbatch 1]);
l           = repmat({':'}, 1, length(Tlbatch));
r           = repmat({':'}, 1, length(Trbatch));
t           = zeros([Tlbatch 1 Trbatch 1], class(x(1)*H(1)));
k           = K+1;
for i=1:K
    t = t + (x(l{:},i,r{:}).^2) .* H(l{:},i,r{:});
    for j=i+1:K
        t = t + x(l{:},i,r{:}) .* x(l{:},j,r{:}) .* H(l{:},k,r{:}) * 2;
        k = k + 1;
    end
end
end