function varargout = spmb_estatics2full(varargin)
% Convert a sparse ESTATICS Hessian to a full matrix
%
% FORMAT F = spmb_estatics2full(H)
% 
% H - (K2 x 1)  Batch of sparse matrices (K2 = 2*K+1)
% F - (K1 x K1) Batch of full matrices
%__________________________________________________________________________
%
% FORMAT spmb_estatics2full(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           H should have shape      (K2 x ...)
%           F   will have shape (K1 x K1 x ...)
%
%      * If `-1`:
%           H should have shape (... x K2)
%           F   will have shape (... x K1 x K1)
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
    [varargout{1:nargout}] = left_2full(dim,args{:});
else
    [varargout{1:nargout}] = right_2full(dim,args{:});
end

end

function F = estatics2full(d,H)
K2                   = size(H,d);
K                    = (K2-1)/2;
shape                = size(H);
lbatch               = shape(1:d-1);
rbatch               = shape(d+1:end);
l                    = repmat({':'}, 1, length(lbatch));
r                    = repmat({':'}, 1, length(rbatch));
F                    = zeros([lbatch K+1 K+1 rbatch]);
iD                   = [l {1:K+1}     r];
iB                   = [l {K+2:2*K+1} r];
F                    = spmb_setdiag(F, H(iD{:}), 'dim', d);
F(l{:},1:K,K+1,r{:}) = spm_unsqueeze(H(iB{:}),d+1);
F(l{:},K+1,1:K,r{:}) = spm_unsqueeze(H(iB{:}),d);
end

function F = left_2full(d,H)
asrow = isrow(H);
if asrow, H = reshape(H, size(H,2), size(H,1)); end
          F = estatics2full(d,H);
if asrow, F = reshape(F, size(F,2), size(F,1)); end
end

function F = right_2full(d,H)
d = ndims(H)+d+1;
ascol = iscolumn(H);
if ascol, H = reshape(H, size(H,2), size(H,1)); end
          F = estatics2full(d,H);
if ascol, F = reshape(F, size(F,2), size(F,1)); end
end