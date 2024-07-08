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
%__________________________________________________________________________
%
% K2 = (K*(K+1))/2
% N2 = (N*(N+1))/2
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
%      * If `-1`:
%           X should have shape (... x K x N)
%           H should have shape (... x N2)
%           A   will have shape (... x K2)
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
    [varargout{1:nargout}] = left_outer(dim,args{:});
else
    [varargout{1:nargout}] = right_outer(dim,args{:});
end

end


% =========================================================================
function idx = mapidx(K)
idx = zeros(K, 'uint64');
k = K+1;
for i=1:K
    idx(i,i) = i;
    for j=i+1:K 
        idx(i,j) = k;
        idx(j,i) = k;
        k = k + 1;
    end
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
function A = left_outer(d,X,H)
K           = size(X,d);
N           = size(X,d+1);
K2          = (K*(K+1))/2;
N2          = (N*(N+1))/2;
Xshape      = size(X);
Xlbatch     = Xshape(1:d-1);
Xrbatch     = Xshape(d+2:end);
if nargin > 2
    if size(H,d) ~= N2
        msg = 'Inconsistant matrix sizes.';
        if isrow(X) || isrow(H)
            msg = [msg ' ' 'Some inputs are row vectors that maybe ' ...
                           'should have been column vectors.'];
        end
        error(msg);
    end
    Hshape   = size(H);
    Hlbatch  = Hshape(1:d-1);
    Hrbatch  = Hshape(d+1:end);
    Xrbatch  = [Xrbatch ones(1,max(0, length(Hrbatch)-length(Xrbatch)))];
    Hrbatch  = [Hrbatch ones(1,max(0, length(Xrbatch)-length(Hrbatch)))];
    Albatch  = max(Xlbatch,Hlbatch);
    Arbatch  = max(Xrbatch,Hrbatch);
else
    Albatch  = Xlbatch;
    Arbatch  = Xrbatch;
end
l           = repmat({':'}, 1, length(Albatch));
r           = repmat({':'}, 1, length(Arbatch));
if nargin > 2
    A       = zeros([Albatch K2 Arbatch 1]);
    Aidx    = mapidx(K);
    for i=1:K
        Xi = reshape(X(l{:},i,:,r{:}), [Xlbatch N Xrbatch]);
        A(l{:},Aidx(i,i),r{:}) = spmb_sym_inner(Xi,H,'dim', d);
    for j=i+1:K
        Xj = reshape(X(l{:},j,:,r{:}), [Xlbatch N Xrbatch]);
        A(l{:},Aidx(i,j),r{:}) = spmb_sym_inner(Xi,H,Xj,'dim', d);
    end
    end
else
    Aidx    = imapidx(K);
    A       = spm_squeeze(dot( ...
                X(l{:},Aidx(:,1),:,r{:}), ...
                X(l{:},Aidx(:,2),:,r{:}), ...
              d+1),d+1);
end
end

% =========================================================================
function A = right_outer(d,X,H)
K           = size(X,ndims(X)+d);
N           = size(X,ndims(X)+d+1);
K2          = (K*(K+1))/2;
N2          = (N*(N+1))/2;
Xshape      = size(X);
Xlbatch     = Xshape(1:end+d-1);
Xrbatch     = Xshape(end+d+2:end);
if nargin > 2
    if size(H,ndims(X)+d+1) ~= N2
        msg = 'Inconsistant matrix sizes.';
        if iscolumn(X) || iscolumn(H)
            msg = [msg ' ' 'Some inputs are columns vectors that ' ...
                           'maybe should have been row vectors.'];
        end
        error(msg);
    end
    Hshape  = size(H);
    Hlbatch = Hshape(1:end+d);
    Hrbatch = Hshape(end+d+2:end);
    Xlbatch = [ones(1,max(0, length(Hlbatch)-length(Xlbatch))) Xlbatch];
    Hlbatch = [ones(1,max(0, length(Xlbatch)-length(Hlbatch))) Hlbatch];
    Xrbatch = [ones(1,max(0, length(Hrbatch)-length(Xrbatch))) Xrbatch];
    Hrbatch = [ones(1,max(0, length(Xrbatch)-length(Hrbatch))) Hrbatch];
    Albatch = max(Xlbatch,Hlbatch);
    Arbatch = max(Xrbatch,Hrbatch);
    X       = reshape(X, [Xlbatch K N Xrbatch]);
    H       = reshape(H, [Hlbatch N2  Hrbatch]);
else
    Albatch = Xlbatch;
    Arbatch = Xrbatch;
end
d           = length(Albatch)+1;
l           = repmat({':'}, 1, length(Albatch));
r           = repmat({':'}, 1, length(Arbatch));
if nargin > 2
    A       = zeros([Albatch K2 Arbatch 1]);
    Aidx    = mapidx(K);
    for i=1:K
        Xi = reshape(X(l{:},i,:,r{:}), [Xlbatch N Xrbatch]);
    for j=i:K
        Xj = reshape(X(l{:},j,:,r{:}), [Xlbatch N Xrbatch]);
        size(Xi)
        size(Xj)
        size(H)
        A(l{:},Aidx(i,j),r{:}) = spmb_sym_inner(Xi,H,Xj,'dim', d);
    end
    end
else
    Aidx   = imapidx(K);
    A       = spm_squeeze(dot( ...
                X(l{:},Aidx(:,1),:,r{:}), ...
                X(l{:},Aidx(:,2),:,r{:}), ...
              d+1),d+1);
end
end