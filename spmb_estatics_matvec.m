function varargout = spmb_estatics_matvec(varargin)
% Product between an ESTATICS matrix and a vector. 
%
% FORMAT v = spmb_estatics_matvec(H,g)
%
% H - (2*K+1) Batch of sparse matrices
% g - (K+1)   Batch of input vectors
% v - (K+1)   Batch of output vectors
%__________________________________________________________________________
%
% FORMAT spmb_estatics_matvec(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           H should have shape (K2 x ...)
%           g should have shape (K1 x ...)
%           v   will have shape (K1 x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           H should have shape (... x K2)
%           g should have shape (... x K1)
%           v   will have shape (... x K1)
%           Batch dimensions are implicitely padded on the left
%__________________________________________________________________________
%
% An ESTATICS matrix has the form H = [D  b; b' r]
% where D = diag(d) is diagonal, b is a vector and r is a scalar.
% It is stored in a flattened form: [d0, d1, ..., dK, r, b0, b1, ...,  bK]
%
% Vectors should be ordered as: [b0, b1, ...,  bK, r]
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_matvec(dim,args{:});
else
    [varargout{1:nargout}] = right_matvec(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (K x ...)
function V = left_matvec(d,H,G)

asrow = isrow(G) && isrow(H);
if isrow(G), G = reshape(G, size(G,2), size(G,1)); end
if isrow(H), H = reshape(H, size(H,2), size(H,1)); end
             V = matvec(d,H,G);
if asrow,    V = reshape(V,size(V,2),size(V,1));   end

end

% =========================================================================
% Python-style: matrix on the right (... x K)
function V = right_matvec(d,H,G)

ascolumn = iscolumn(G) && iscolumn(H);
if iscolumn(G), G = reshape(G, size(G,2), size(G,1)); end
if iscolumn(H), H = reshape(H, size(H,2), size(H,1)); end
                V = matvec(d,H,G);
if ascolumn,    V = reshape(V,size(V,2),size(V,1));   end

end

% =========================================================================
% Generic implementation
function V = matvec(d,H,G)

% Pad dimensions
[Glbatch,K,Grbatch] = spmb_splitsize(G,d,1);
[Hlbatch,~,Hrbatch] = spmb_splitsize(H,d,1);
[Glbatch,Hlbatch]   = spmb_pad_shapes(Glbatch,Hlbatch,'L');
[Grbatch,Hrbatch]   = spmb_pad_shapes(Grbatch,Hrbatch,'R');
Vlbatch             = max(Glbatch,Hlbatch);
Vrbatch             = max(Grbatch,Hrbatch);

K = K - 1;
G = reshape(G, [Glbatch K+1   Grbatch 1]);
H = reshape(H, [Hlbatch 2*K+1 Hrbatch 1]);
V = zeros([Vlbatch K+1 Vrbatch 1], class(H(1)*G(1)));
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

% Compute matrix-vector product
V(iD{:}) =     D .* Gb     + B .* Gr;
V(iR{:}) = dot(B,   Gb, d) + R .* Gr;

end