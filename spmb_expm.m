function varargout = spmb_expm(varargin)
% Approximate matrix exponential using a Taylor expansion
%
% !!! very inefficient implementation due to inefficient spmb_lmdiv
%
% FORMAT E = spmb_expm(J)
%
% J - (N x N)  Input batch of matrices
% E - (N x N)  Output batch of exponentiated matrices: E = expm(J)
%
% FORMAT y = spmb_expm(J,x)
%
% J - (N x N)  Input batch of matrices
% x - (N x 1)  Input batch of vectors
% y - (N x 1)  Output matrix-vector product: y = expm(J) * x
%__________________________________________________________________________
%
% FORMAT spmb_expm(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           J should have shape (N x N x ...)
%           E   will have shape (N x N x ...)
%           x should have shape     (N x ...)
%           y   will have shape     (N x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           J should have shape (... x N x N)
%           E   will have shape (... x N x N)
%           x should have shape (... x N x 1)
%           y   will have shape (... x N x 1)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
%__________________________________________________________________________
%
% This function follows Karl's spm_expm implementation.
%__________________________________________________________________________

% Yael Balbastre, Karl Friston

[dim,args] = spmb_parse_dim(varargin{:});
[varargout{1:nargout}] = do_expm(dim,args{:});

end

% =========================================================================
function N = norm_inf(X,d)
    % Matrix infinity norm - preserves reduced dimensions
    if d > 0
        N = max(sum(abs(X),d+1),[], d);
    else
        N = max(sum(abs(X),ndims(X)+d+1),[], ndims(X)+d);
    end
end

% =========================================================================
function E = do_expm(d,J,x)
if nargin == 3
    E = spmb_matvec(do_batch_expm(dim,J),x,'dim',d);
    return
end

if d > 0
    N = size(J,d);
    L = d-1;
else
    N = size(J,ndims(J)+d);
    L = ndims(J)+d-1;
end

% ensure norm is < 1/2 by scaling by power of 2
%--------------------------------------------------------------------------
I     = spm_unsqueeze(eye(N),1,L);
[~,e] = log2(norm_inf(J,d));
s     = max(0,e + 1);
J     = J./2.^s;
X     = J;
c     = 1/2;
E     = I + c*J;
D     = I - c*J;
q     = 6;
p     = 1;
for k = 2:q
    c   = c*(q - k + 1)/(k*(2*q - k + 1));
    X   = spmb_matmul(J, X, 'dim', d);
    cX  = c*X;
    E   = E + cX;
    if p
        D = D + cX;
    else
        D = D - cX;
    end
    p = ~p;
end

% E = inv(D)*E
%--------------------------------------------------------------------------
E = spmb_lmdiv(D, E, 'dim', d);

% Undo scaling by repeated squaring E = E^(2^s)
%--------------------------------------------------------------------------
if d == 1
    s = spm_squeeze(spm_squeeze(s,2),1);
    for k = 1:max(s)
        E1 = E(:,:,k<=s);
        E(:,:,k<=s) = spmb_matmul(E1, E1, 'dim', d);
    end
elseif d == -1
    n = ndims(s);
    s = spm_squeeze(spm_squeeze(s,n),n-1);
    for k = 1:max(s)
        E1 = E(k<=s,:,:);
        E(k<=s,:,:) = spmb_matmul(E1, E1, 'dim', d);
    end
else
    if d < 0, d = ndims(E) + d; end
    E = spmb_movedim(E,d+1,1);
    E = spmb_movedim(E,d,1);
    s = spm_squeeze(s,[d d+1]);
    for k = 1:max(s)
        E1 = E(:,:,k<=s);
        E(:,:,k<=s) = spmb_matmul(E1, E1, 'dim', d);
    end
    E = spmb_movedim(E,1,d);
    E = spmb_movedim(E,1,d+1);
end
end