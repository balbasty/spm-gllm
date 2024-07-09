function W = gllm_reml_mask(Y,X)
% Automatically select voxels that enter covariance estimate
%
% FORMAT W = gllm_reml_mask(Y,X)
%
% Y - (N... x M) Observations
% X - (M x K)    Design matrix
% W - (N... x 1) Mask of voxels to include
%__________________________________________________________________________

% Yael Balbastre

% -------------------------------------------------------------------------
% Checks sizes

M      = size(X,1);
Yshape = size(Y);
if M == 1, Nd = ndims(Y); if Nd == 2 && size(Y,2) == 1, Nd = 1; end
else,      Nd = ndims(Y)-1; end
if Nd > 1
    Nz = size(Y,Nd);
else
    Nz = 1;
    M  = size(Y,2);
end

if size(Y,Nd+1) ~= M
    error('Incompatible matrix sizes')
end

% Build a cell of handles that load slices on the fly
Y0 = Y;
if Nd > 1, Y = cellfun(@(n) (@() reshape(spm_subsref(Y0,{n},Nd), [], M)), num2cell(1:Nz), 'UniformOutput', false);
else,      Y = {@() Y0};  end

% ---------------------------------------------------------------------
% Run NLS fit + Compute residuals
V = zeros(Yshape(1:Nd));  % full variance
R = zeros(Yshape(1:Nd));  % explained variance
for z=1:Nz
    Yz      = Y{z}();
    Rz      = zeros(size(Yz,1),1);
    msk     = all(isfinite(Yz) & Yz ~= 0,2);
    Vz      = mean(Yz.*Yz,2) - mean(Yz,2).^2;
    Vz      = reshape(Vz, [Yshape(1:Nd-1) 1]);
    V       = spm_subsasgn(V,Vz,{z},Nd);
    Yz      = Yz(msk,:);
    Bz      = gllm_fit(Yz,X,1);
    Rz(msk) = residuals(Yz,X,Bz);
    Rz      = reshape(Rz, [Yshape(1:Nd-1) 1]);
    R       = spm_subsasgn(R,Rz,{z},Nd);
end


% ---------------------------------------------------------------------
% Compute F statistics + threshold
F = V ./ R - 1;
F(R == 0) = NaN;
t = spm_invFcdf(1 - 0.05,[M M]);
W = (F > t) & (R < 3*median(R(:)));
end

% =========================================================================
function R = residuals(Y,X,B)
Z = exp(B*X');
R = Z - Y;
R = dot(R,R,2) / size(X,2);
end

function XAX = outer(A,X)
Xt  = spm_unsqueeze(X',1);
X   = spm_unsqueeze(X,1);
XA  = spmb_matmul(X,A,2);
XAX = spmb_matmul(XA,Xt,2);
end