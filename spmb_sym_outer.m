function varargout = spmb_sym_outer(varargin)
% Symmetric outer product X*H*X' stored in a sparse form
%
% FORMAT A = spmb_sym_outer(X)
% 
% X - (K  x N)  Batch of input matrices
% A - (K2 x 1)  Batch of output sparse matrices: A = X*X'
%
% FORMAT A = spmb_sym_outer(X,H)
% 
% X - (K  x N)  Batch of input matrices
% H - (N2 x 1)  Batch of input sparse matrices
% A - (K2 x 1)  Batch of output sparse matrices: A = X*H*X'
%
% FORMAT A = spmb_sym_outer(X,H,Y)
% 
% X - (K  x N)  Batch of input matrices
% H - (N2 x 1)  Batch of input sparse matrices
% Y - (K  x N)  Batch of input matrices
% A - (K2 x 1)  Batch of output sparse matrices: A = X*H*Y' + Y*H*X'
%__________________________________________________________________________
%
% K2 = (K*(K+1))/2
% N2 = (N*(N+1))/2 , if symmetric or 
%       N          , if diagonal  or
%       0          , if identity  .
%__________________________________________________________________________
%
% FORMAT spmb_sym_outer(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           X should have shape   (K x N x ...)
%           H   should have shape    (N2 x ...)
%           A   will have shape      (K2 x ...)
%
%      * If `2`:
%           X should have shape   (B x K x N  x ...)
%           H   should have shape (B x     N2 x ...)
%           A   will have shape   (B x     K2 x ...)
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

if length(args) == 1 || (length(args) == 2 && isempty(args{2}))
    [varargout{1:nargout}] = outer_XX(dim,args{:});
elseif length(args) == 2
    [varargout{1:nargout}] = outer_XHX(dim,args{:});
elseif length(args) == 3 && isempty(args{2})
    args(2) = [];
    [varargout{1:nargout}] = outer_XY(dim,args{:});
else
    [varargout{1:nargout}] = outer_XHY(dim,args{:});
end

end

% =========================================================================
function idx = imapidx(K)
K2 = (K*(K+1))/2;
idx = zeros(K2, 2, 'uint64');
k = K+1;
for i=1:K
    idx(i,1) = i;
    idx(i,2) = i;
    for j=i+1:K 
        idx(k,1) = i;
        idx(k,2) = j;
        k = k + 1;
    end
end
end

% =========================================================================
function A = outer_XHX(d,X,H)
K  = size(X,d);
N  = size(X,d+1);
K2 = (K*(K+1))/2;
N2 = (N*(N+1))/2;

if ~any(size(H,d) == [N N2])
    msg = 'Inconsistant matrix sizes.';
    if isrow(X) || isrow(H)
        msg = [msg ' ' 'Some inputs are row vectors that maybe ' ...
                       'should have been column vectors.'];
    end
    error(msg);
end

Xshape  = size(X);
Xlbatch = Xshape(1:d-1);
Xrbatch = Xshape(d+2:end);
Hshape  = size(H);
Hlbatch = Hshape(1:d-1);
Hrbatch = Hshape(d+1:end);
Xrbatch = [Xrbatch ones(1,max(0, length(Hrbatch)-length(Xrbatch)))];
Hrbatch = [Hrbatch ones(1,max(0, length(Xrbatch)-length(Hrbatch)))];
Albatch = max(Xlbatch,Hlbatch);
Arbatch = max(Xrbatch,Hrbatch);
l       = repmat({':'}, 1, length(Albatch));
r       = repmat({':'}, 1, length(Arbatch));
A       = zeros([Albatch K2 Arbatch 1]);
k       = K+1;
for i=1:K
    Xi             = spm_squeeze(X(l{:},i,:,r{:}), d);
    A(l{:},i,r{:}) = spmb_sym_inner(Xi,H,'dim', d);
for j=i+1:K
    Xj             = spm_squeeze(X(l{:},j,:,r{:}), d);
    A(l{:},k,r{:}) = spmb_sym_inner(Xi,H,Xj,'dim', d);
    k              = k + 1;
end
end
end

% =========================================================================
function A = outer_XX(d,X)
K          = size(X,d);
Xshape     = size(X);
Xlbatch    = Xshape(1:d-1);
Xrbatch    = Xshape(d+2:end);
Albatch    = Xlbatch;
Arbatch    = Xrbatch;
l          = repmat({':'}, 1, length(Albatch));
r          = repmat({':'}, 1, length(Arbatch));
Aidx       = imapidx(K);
A          = spm_squeeze(dot(X(l{:},Aidx(:,1),:,r{:}), ...
                             X(l{:},Aidx(:,2),:,r{:}), ...
             d+1),d+1);
end

% =========================================================================
function A = outer_XY(d,X,Y)
K          = size(X,d);
N          = size(X,d+1);
if size(Y,d) ~= K || size(Y,d+1) ~= N
    msg = 'Inconsistant matrix sizes.';
    if isrow(X) || isrow(Y)
        msg = [msg ' ' 'Some inputs are row vectors that maybe ' ...
                       'should have been column vectors.'];
    end
    error(msg);
end
Xshape     = size(X);
Xlbatch    = Xshape(1:d-1);
Xrbatch    = Xshape(d+2:end);
Yshape     = size(X);
Ylbatch    = Yshape(1:d-1);
Yrbatch    = Yshape(d+2:end);
Xrbatch    = [Xrbatch ones(1,max(0,length(Yrbatch)-length(Xrbatch)))];
Yrbatch    = [Yrbatch ones(1,max(0,length(Xrbatch)-length(Yrbatch)))];
Albatch    = max(Xlbatch,Ylbatch);
Arbatch    = max(Xrbatch,Yrbatch);
l          = repmat({':'}, 1, length(Albatch));
r          = repmat({':'}, 1, length(Arbatch));
Aidx       = imapidx(K);
A          = spm_squeeze(dot(X(l{:},Aidx(:,1),:,r{:}), ...
                             Y(l{:},Aidx(:,2),:,r{:}), ...
             d+1),d+1);
A          = A + spm_squeeze(dot(Y(l{:},Aidx(:,2),:,r{:}), ...
                                 X(l{:},Aidx(:,1),:,r{:}), ...
             d+1),d+1);
end


% =========================================================================
function A = outer_XHY(d,X,H,Y)
K  = size(X,d);
N  = size(X,d+1);
K2 = (K*(K+1))/2;
N2 = (N*(N+1))/2;

if ~any(size(H,d) == [N N2]) || size(Y,d) ~= K || size(Y,d+1) ~= N
    msg = 'Inconsistant matrix sizes.';
    if isrow(X) || isrow(H)
        msg = [msg ' ' 'Some inputs are row vectors that maybe ' ...
                       'should have been column vectors.'];
    end
    error(msg);
end

Xshape  = size(X);
Xlbatch = Xshape(1:d-1);
Xrbatch = Xshape(d+2:end);
Yshape  = size(X);
Ylbatch = Yshape(1:d-1);
Yrbatch = Yshape(d+2:end);
Hshape  = size(H);
Hlbatch = Hshape(1:d-1);
Hrbatch = Hshape(d+1:end);
nrbatch = max([length(Hrbatch) length(Xrbatch) length(Yrbatch)]);
Xrbatch = [Xrbatch ones(1,max(0, nrbatch-length(Xrbatch)))];
Yrbatch = [Yrbatch ones(1,max(0, nrbatch-length(Yrbatch)))];
Hrbatch = [Hrbatch ones(1,max(0, nrbatch-length(Hrbatch)))];
Albatch = max(max(Xlbatch,Hlbatch),Ylbatch);
Arbatch = max(max(Xrbatch,Hrbatch),Yrbatch);
l       = repmat({':'}, 1, length(Albatch));
r       = repmat({':'}, 1, length(Arbatch));
A       = zeros([Albatch K2 Arbatch 1]);
k       = K+1;
for i=1:K
    Xi             = spm_squeeze(X(l{:},i,:,r{:}), d);
    Yi             = spm_squeeze(Y(l{:},i,:,r{:}), d);
    A(l{:},i,r{:}) = spmb_sym_inner(Xi,H,Yi,'dim', d) * 2;
for j=i+1:K
    Xj             = spm_squeeze(X(l{:},j,:,r{:}), d);
    Yj             = spm_squeeze(Y(l{:},j,:,r{:}), d);
    A(l{:},k,r{:}) = spmb_sym_inner(Xi,H,Yj,'dim', d) ...
                   + spmb_sym_inner(Yi,H,Xj,'dim', d);
    k              = k + 1;
end
end
end