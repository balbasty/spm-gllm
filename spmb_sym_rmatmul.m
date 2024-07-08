function Y = spmb_sym_rmatmul(X,A,varargin)
% Matrix multiplication, where the right matrix is compact symmetric
%
% FORMAT Y = spmb_sym_rmatmul(X,A)
% 
% X -  (M x N)  Batch of input vectors
% A - (N2 x 1)  Batch of input compact symmetric matrices
% Y -  (N x N)  Batch of output vectors
%__________________________________________________________________________
%
% N2 = (N*(N+1))/2
%__________________________________________________________________________
%
% FORMAT spmb_sym_rmatmul(A,X,DIM)
% FORMAT spmb_sym_rmatmul(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape      (N2 x ...)
%           X should have shape   (M x N x ...)
%           Y   will have shape   (M x N x ...)
%
%      * If `-1`:
%           A should have shape (... x N2)
%           X should have shape (... x M x N)
%           Y   will have shape (... x M x N)
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
    Y = left_matmul(dim,X,A);
else
    Y = right_matmul(dim,X,A);
end
end

% =========================================================================
function Y = left_matmul(d,X,A)
M           = size(X,d);
N           = size(X,d+1);
N2          = (N*(N+1))/2;
if size(A,d) ~= N2
    msg = 'Inconsistant matrix sizes.';
    if isrow(X) || isrow(A)
        msg = [msg ' ' 'Some inputs are row vectors that maybe should ' ...
                       'have been column vectors.'];
    end
    error(msg);
end
Xshape      = size(X);
Xlbatch     = Xshape(1:d-1);
Xrbatch     = Xshape(d+2:end);
Ashape      = size(A);
Albatch     = Ashape(1:d-1);
Arbatch     = Ashape(d+1:end);
nbatch      = max([length(Arbatch) length(Xrbatch)]);
Xrbatch     = [Xrbatch ones(1,nbatch-length(Xrbatch))];
Arbatch     = [Arbatch ones(1,nbatch-length(Arbatch))];
Yrbatch     = max(Xrbatch,Arbatch);
Ylbatch     = max(Xlbatch,Albatch);
Y           = zeros([Ylbatch M N Yrbatch]);
A           = spm_unsqueeze(A,d-1);
l           = repmat({':'}, 1, length(Ylbatch));
r           = repmat({':'}, 1, length(Yrbatch));
k           = N+1;
for i=1:N
    Y(l{:},:,i,r{:}) = Y(l{:},:,i,r{:}) + X(l{:},:,i,r{:}) .* A(l{:},:,i,r{:});
    for j=i+1:N
        Y(l{:},:,j,r{:}) = Y(l{:},:,j,r{:}) + X(l{:},:,i,r{:}) .* A(l{:},:,k,r{:});
        Y(l{:},:,i,r{:}) = Y(l{:},:,i,r{:}) + X(l{:},:,j,r{:}) .* A(l{:},:,k,r{:});
        k = k + 1;
    end
end
end

% =========================================================================
function Y = right_matmul(d,X,A)
M           = size(X,ndims(X)+d);
N           = size(X,ndims(X)+d+1);
N2          = (N*(N+1))/2;
if size(A,ndims(A)+d+1) ~= N2
    msg = 'Inconsistant matrix sizes.';
    if iscolumn(X) || iscolumn(A)
        msg = [msg ' ' 'Some inputs are columns vectors that maybe ' ...
                       'should have been row vectors.'];
    end
    error(msg);
end
Xshape      = size(X);
Xlbatch     = Xshape(1:end+d-1);
Xrbatch     = Xshape(end+d+2:end);
Ashape      = size(A);
Albatch     = Ashape(1:end+d);
Arbatch     = Ashape(end+d+2:end);
nbatch      = max([length(Albatch) length(Xlbatch)]);
Xlbatch     = [ones(1,nbatch-length(Xlbatch)) Xlbatch];
Albatch     = [ones(1,nbatch-length(Albatch)) Albatch];
Yrbatch     = max(Xrbatch,Arbatch);
Ylbatch     = max(Xlbatch,Albatch);
X           = reshape(X, [Xlbatch M N Xrbatch]);
A           = reshape(A, [Albatch N2 1 Arbatch]);
Y           = zeros([Ylbatch 1 Yrbatch 1]);
l           = repmat({':'}, 1, length(Ylbatch));
r           = repmat({':'}, 1, length(Yrbatch));
k           = N+1;
for i=1:N
    Y(l{:},:,i,r{:}) = Y(l{:},:,i,r{:}) + X(l{:},:,i,r{:}) .* A(l{:},:,i,r{:});
    for j=i+1:N
        Y(l{:},:,j,r{:}) = Y(l{:},:,j,r{:}) + X(l{:},:,i,r{:}) .* A(l{:},:,k,r{:});
        Y(l{:},:,i,r{:}) = Y(l{:},:,i,r{:}) + X(l{:},:,j,r{:}) .* A(l{:},:,k,r{:});
        k = k + 1;
    end
end
end