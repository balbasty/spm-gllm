function varargout = spmb_estatics_solve(varargin)
% Left-matrix division between an ESTATICS matrix and a vector
%
% FORMAT v = spmb_estatics_solve(H,g)
% FORMAT v = spmb_estatics_solve(H,g,l)
%
% H - (2*K+1 x 1) Batch of sparse matrices
% g - (K+1   x 1) Batch of input vectors
% l - (1|K+1 x 1) Batch of loading vector(s)
% v - (K+1   x 1) Batch of output vectors
%__________________________________________________________________________
%
% FORMAT spmb_estatics_solve(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           H should have shape (K2 x ...)
%           g should have shape (K1 x ...)
%           l should have shape (?  x ...)
%           v   will have shape (K1 x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           H should have shape (... x K2)
%           g should have shape (... x K1)
%           l should have shape (... x ?)
%           v   will have shape (... x K1)
%           Batch dimensions are implicitely padded on the left
%__________________________________________________________________________
%
% An ESTATICS matrix has the form H = [D  b; b' r]
% where D = diag(d) is diagonal, b is a vector and r is a scalar.
% It is stored in a flattened form: [d0, d1, ..., dK, r, b0, b1, ...,  bK]
%
% Because of this specific structure, the Hessian is inverted in
% closed-form using the formula for the inverse of a 2x2 block matrix.
% See: https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
%
% Vectors should be ordered as: [b0, b1, ...,  bK, r]
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_solve(dim,args{:});
else
    [varargout{1:nargout}] = right_solve(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (K x ...)
function V = left_solve(d,H,G,L)

if nargin < 4, L = 0; end

K  = size(G,d);
K2 = size(H,d);
Q  = size(L,d);

if K2 ~= 2*K-1 || (Q ~= K && Q ~= 1)
    msg = 'Incompatible shapes.';
    if isvector(G) || isvector(H) || isvector(L)
        msg = [msg ' ' 'Some inputs may be row vectors instead of ' ...
                       'column vectors or vice versa.'];
    end
    error(msg);
end

V = solve(d,H,G,L);
end

% =========================================================================
% Python-style: matrix on the right (... x K)
function V = right_solve(d,H,G,L)

if nargin < 4, L = 0; end

K  = size(G,ndims(G)+d+1);
K2 = size(H,ndims(H)+d+1);
Q  = size(L,ndims(L)+d+1);

if K2 ~= 2*K-1 || (Q ~= K && Q ~= 1)
    msg = 'Incompatible shapes.';
    if isvector(G) || isvector(H) || isvector(L)
        msg = [msg ' ' 'Some inputs may be row vectors instead of ' ...
                       'column vectors or vice versa.'];
    end
    error(msg);
end

V = solve(d,H,G,L);
end

% =========================================================================
% Generic implementation
function V = solve(d,H,G,L)

% Pad dimensions
[Glbatch,K,Grbatch] = spmb_splitsize(G,d,1);
[Hlbatch,~,Hrbatch] = spmb_splitsize(H,d,1);
[Llbatch,Q,Lrbatch] = spmb_splitsize(L,d,1);
[Glbatch,Hlbatch,Llbatch] = spmb_pad_shapes(Glbatch,Hlbatch,Llbatch,'L');
[Grbatch,Hrbatch,Lrbatch] = spmb_pad_shapes(Grbatch,Hrbatch,Lrbatch,'R');
Vlbatch = max(max(Glbatch,Hlbatch),Llbatch);
Vrbatch = max(max(Grbatch,Hrbatch),Lrbatch);

K = K - 1;
L = reshape(L, [Llbatch     Q  Lrbatch 1]);
G = reshape(G, [Glbatch   K+1  Grbatch 1]);
H = reshape(H, [Hlbatch 2*K+1  Hrbatch 1]);
V = zeros([Vlbatch K+1 Vrbatch 1], class(G(1)/H(1)));
l = repmat({':'}, 1, length(Vlbatch));
r = repmat({':'}, 1, length(Vrbatch));

if d < 0
    d = ndims(V)+d+1;
end

% Extract matrix components
iD = [l {1:K}       r]; D = H(iD{:});         % diagonal
iR = [l {K+1}       r]; R = H(iR{:});         % bottom-right element
iB = [l {K+2:2*K+1} r]; B = H(iB{:});         % off-diagonal

% Extract vector components
Gb = G(iD{:});
Gr = G(iR{:});

% Load hessian
if nargin > 3
    if Q == 1
        D = D + L;
        R = R + L;
    else
        D = D + L(iD{:});
        R = R + L(iR{:});
    end
end

% precompute stuff
vnorm    = B./D;
mini_inv = R - dot(B, vnorm, d);

% top left corner
V(iD{:}) = (dot(vnorm, Gb, d) ./ mini_inv) .* vnorm;
V(iD{:}) = V(iD{:}) + Gb./D;

% top right corner:
V(iD{:}) = V(iD{:}) - vnorm .* Gr ./ mini_inv;

% bottom left corner:
V(iR{:}) = - dot(vnorm, Gb, d) ./ mini_inv;

% bottom right corner:
V(iR{:}) = V(iR{:}) + Gr ./ mini_inv;

end