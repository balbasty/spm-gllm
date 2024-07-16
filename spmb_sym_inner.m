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
%           x should have shape      (K  x ...)
%           y should have shape      (K  x ...)
%           t   will have shape      (1  x ...)
%
%      * If `2`:
%           H should have shape      (B x K2 x ...)
%           x should have shape      (B x K  x ...)
%           y should have shape      (B x K  x ...)
%           t   will have shape      (B x 1  x ...)
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
if length(args) == 2
    [varargout{1:nargout}] = inner_xx(dim,args{:});
else
    [varargout{1:nargout}] = inner_xy(dim,args{:});
end
end

% =========================================================================
function t = inner_xy(d,x,H,y)
if nargin < 4, y = x; end
K           = size(x,d);
K2          = (K*(K+1))/2;
if ~any(size(H,d) == [K K2]) || size(y,d) ~= K
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
    if size(H,d) > K
        for j=i+1:K
            xj = x(l{:},j,r{:});
            yj = y(l{:},j,r{:});
            hk = H(l{:},k,r{:});
            t  = t + (xi .* yj + xj .* yi) .* hk;
            k  = k + 1;
        end
    end
end
end

% =========================================================================
function t = inner_xx(d,x,H)
K           = size(x,d);
K2          = (K*(K+1))/2;
if ~any(size(H,d) == [K K2])
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
    if size(H,d) > K
        for j=i+1:K
            xj = x(l{:},j,r{:});
            hk = H(l{:},k,r{:});
            t  = t + 2 * (xi .* xj) .* hk;
            k  = k + 1;
        end
    end
end
end
