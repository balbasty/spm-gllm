function [F,G,H] = gllm_mc_diff(Y,X,B,W,opt)
% Derivatives of a a sum of loglinear models.
%
% FORMAT [F,G,H] = gllm_mc_diff(Y,X,B,W,opt)
%
% Y - (N x M)       Observations
% X - (M x K x C)   Design matrix
% B - (N x K x C)   Model parameters
% W - (N x M x M)   Precision matrix (= WLS weights) [1]
% F - (N x M)       Signal fits
% G - (N x K)       Gradients
% H - (N x K2)      Hessians
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of parameters
% C - Number of components
%__________________________________________________________________________
%
% opt.accel - (float)         Robustness trade-off  [0]
% opt.mode  - ('full'|'sym')  Hessian mode          [sym]
%__________________________________________________________________________
%
% This function differentiates the mean squared error (or its weighted 
% version) between the observations Y and a fit F
%
%                       0.5 * trace((Y-F)'*(Y-F))
%
% with respect to model parameters B.
%
% In the single component case, the model of the signal is F = exp(B*X').
% 
% Multi component models are made of a sum of log-linear models.
% There are three different ways of defining such a model:
%
% * If the design matrix has shape (M x K x C) with C > 1, the log-signal
%   in each component of the sum is build from a different design matrix
%   that act on the _same_ model parameters:
%
%                   F = \sum_c exp(B * X(:,:,c)')
%
%   This enables complex designs where a set of parameters can act on 
%   multiple compartments at once.
% 
% * If a single design matrix with shape (M x K) is provided, but the
%   model parameters has shape (M x K x C) with `C > 1`, each component 
%   has its own set of parameters, on which act a shared design matrix:
%
%                   F = \sum_c exp(B(:,:,c) * X')
%
% * Finally, both cases can be combined, in which case each model in the
%   sum has its own design matrix and set of parameters:
%
%                   F = \sum_c exp(B(:,:c) * X(:,:,c)')
%
% Approximate Hessian
% -------------------
% The Hessian of a log-linear model is not positive definite everywhere.
% We therefore return a positive definite approximation instead.
% * When accel=1, the Gauss-Newton approximation of the Hessian is
%   returned.
% * When accel=0, the Gausss-Newton Hessian is additionally loaded with a
%   data-dependent term that ensures convergence.
% * One can interpolate between both approximations by setting accel to any
%   value between 0 and 1.
%
% Hessian mode
% ------------
% * If mode is 'full', a batch of full Hessian matrices is returned, 
%   with shape [N K K].
%
% * If mode is 'sym', a compact flattened representation of the Hessian 
%   is returned, with shape [N K*(K+1)/2]. The flattened vector contains
%   the main diagonal of the Hessian followed by each row of its upper
%   triangular part: [H(1,1) H(2,2) H(3,3) H(1,2) H(1,3) H(2,3)]
%
% Precision matrix
% ----------------
% The precision matrix W can have shape:
%
%   (N x M x M)     Block-diagonal (with blocks == voxels)
%   (N x M2)        Block-diagonal, with compact storage
%   (N x M)         Diagonal
%   (N x 1)         Diagonal, shared across volumes
%   (1 x 1)         Scaled identity, shared across volumes and voxels
%   (1 x M)         Diagonal, shared across voxels
%   (1 x M x M)     Shared across voxels
%   (1 x M2)        Shared across voxels, with compact storage
%
% where M2 = (M*(M+1))/2
%__________________________________________________________________________

% Yael Balbastre

% -------------------------------------------------------------------------
% Fallback to simple component
if max(size(X,3),size(B,3)) == 1
    [F,G,H] = gllm_diff(Y,X,B,W,opt);
    return;
end

% -------------------------------------------------------------------------
% Options

if nargin < 4, W   = 1;      end
if nargin < 5, opt = struct; end

if ~isfield(opt, 'accel'), opt.accel = 0;      end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto'; end

% -------------------------------------------------------------------------
% Checks sizes

N  = size(Y,1);
M  = size(X,1);
K  = size(X,2);
Cx = size(X,3);
Cb = size(B,3);
C  = max(Cx,Cb);

if size(Y,2) ~= M                       ...
|| size(B,1) ~= N                       ...
|| size(B,2) ~= K                       ...
|| ~any(size(W,1) == [1 N])             ...
|| ~any(size(W,2) == [1 M (M*(M+1))/2]) ...
|| ~any(size(W,3) == [1 M])             ...
|| ~any(Cx        == [1 C])             ...
|| ~any(Cb        == [1 C])
    error('Incompatible matrix sizes')
end

% -------------------------------------------------------------------------
% Set precision and Hessian layout
if ~ismatrix(W),        wmode = 'F';    % Full
elseif size(W,2) > M,   wmode = 'S';    % Compact symmetric
else,                   wmode = 'D';    % Compact diagonal
end

if wmode == 'F', W = spmb_full2sym(W,'dim',2); end

hmode = upper(opt.mode(1));

% -------------------------------------------------------------------------
% Fitted signal
Fc = zeros([N M Cb], class(B));
for c=1:C
    Fc(:,:,c) = exp(B(:,:,min(c,Cb))*X(:,:,min(c,Cx))');
end
F = sum(Fc,3);

if nargout == 1, return; end

% -------------------------------------------------------------------------
% Residuals
R = F - Y;

% -------------------------------------------------------------------------
% Gradient
FR = smart_gradient(W, Fc .* R, wmode);
G  = spmb_matmul(FR, X);
if Cb == 1, G = sum(G,3); end

if nargout == 2, return; end

% -------------------------------------------------------------------------
% Hessian
H = smart_hessian(W, Fc, X, wmode, Cb);
if opt.accel ~= 1
    AFR = spm_unsqueeze(abs(FR),3);
    Xt  = spm_unsqueeze(permute(X, [2 1 3]),[1 3]);
    AFR = spmb_sym_outer(Xt,AFR,'dim',2);
    AFR = spm_squeeze(AFR,3);
    if Cb == 1
        AFR = sum(AFR,3);
    end
    H = H + AFR * (1 - opt.accel);
end

if hmode == 'F', H = spmb_sym2full(H); end

% =========================================================================
function H = smart_hessian(W,F,X,wmode,Cb)
% Compute X(:,:,c)' * ((F(:,:,c)*F(:,:,d)') .* W) * X(:,:,d) 
% for each pair of (c,d). If Cb == 1, sum across all pairs of (c,d).
% -------------------------------------------------------------------------
% Sizes
N    = size(F,1);
M    = size(F,2);
C    = size(F,3);
K    = size(X,2);
Cx   = size(X,3);
KC   = K*Cb;
% -------------------------------------------------------------------------
if Cb == 1 && wmode == 'D'
    H = zeros([N K*(K+1)/2]);
    for c=1:C
        Xc = X(:,:,min(Cx,c));
        Xc = spm_unsqueeze(Xc',1);
        Fc = F(:,:,c);
        H  = H + spmb_sym_outer(Xc,(Fc.^2 .* W),'dim',2);
        for d=c+1:C
            Xd = X(:,:,min(Cx,d));
            Xd = spm_unsqueeze(Xd',1);
            Fd = F(:,:,d);
            H = H + spmb_sym_outer(Xc,(Fc.*Fd.*W),Xd,'dim',2);
        end
    end
% -------------------------------------------------------------------------
elseif Cb == 1 && wmode ~= 'D'
    H = zeros([N (K*(K+1))/2]);
    for c=1:C
        Xc = X(:,:,min(Cx,c));
        Fc = F(:,:,c);
        WW = spmb_sym_outer(Fc,'dim',2) .* W;
        H  = H + spmb_sym_outer(Xc,WW,'dim',2);
        for d=c+1:C
            Xd = X(:,:,min(Cx,d));
            Fd = F(:,:,d);
            km = M+1;
            for m=1:M
                WW = (Fc(:,m) .* Fc(:,m)) .* W(:,m);
                H  = H + spmb_sym_outer(Xc(m,:),Xd(m,:),'dim',2) .* WW;
                for p=m+1:M
                    WW = (Fc(:,m) .* Fd(:,p)) .* W(:,km);
                    H  = H + spmb_sym_outer(Xc(m,:),Xd(p,:),'dim',2) .* WW;
                    WW = (Fc(:,p) .* Fd(:,m)) .* W(:,km);
                    H  = H + spmb_sym_outer(Xc(p,:),Xd(m,:),'dim',2) .* WW;
                    km = km + 1;
                end
            end
        end
    end
% -------------------------------------------------------------------------
elseif Cb > 1 && wmode == 'D'
    H = zeros([N K C K C]);
    W = sqrt(W);
    for c=1:C
        Xc = X(:,:,min(Cx,c));
        Fc = W .* F(:,:,c);
        for i=1:K
            Xci = Xc(:,i)' .* Fc;
            H(:,i,c,i,c) = dot(Xci,Xci,2);
            for j=i+1:K
                Xcj = Xc(:,j)' .* Fc;
                H(:,i,c,j,c) = dot(Xci,Xcj,2);
                H(:,j,c,i,c) = H(:,i,c,j,c);
            end
        end
        for d=c+1:C
            Xd = X(:,:,min(Cx,d));
            Fd = W .* F(:,:,d);
            for i=1:K
                Xci = Xc(:,i)' .* Fc;
                Xdi = Xd(:,i)' .* Fd;
                H(:,i,c,i,d) = dot(Xci,Xdi,2);
                H(:,i,d,i,c) = H(:,i,c,i,d);
                for j=i+1:K
                    Xcj = Xc(:,j)' .* Fc;
                    Xdj = Xd(:,j)' .* Fd;
                    H(:,i,c,j,d) = dot(Xci,Xdj,2);
                    H(:,j,c,i,d) = dot(Xcj,Xdi,2);
                    H(:,i,d,j,c) = H(:,j,c,i,d);
                    H(:,j,d,i,c) = H(:,i,c,j,d);
                end
            end
        end
    end
% -------------------------------------------------------------------------
elseif Cb > 1 && wmode ~= 'D'
    H = zeros([N K C K C]);
    W = spmb_sym2full(W,'dim',2);
    for c=1:C
        Xc = X(:,:,min(Cx,c));
        Fc = F(:,:,c);
        Wcc = W .* (Fc .* spm_unsqueeze(Fc,2));
        for i=1:K
            Xci = spmb_matmul(Wcc,Xc(:,i)','dim',2);
            H(:,i,c,i,c) = spmb_dot(Xci,Xc(:,i)',2);
            for j=i+1:K
                H(:,i,c,j,c) = spmb_dot(Xci,Xc(:,j)',2);
                H(:,j,c,i,c) = H(:,i,c,j,c);
            end
        end
        for d=c+1:C
            Xd = X(:,:,min(Cx,d));
            Fd = F(:,:,d);
            Wcd = W .* (Fc .* spm_unsqueeze(Fd,2));
            Wdc = spmb_transpose(Wcd,'dim',2);
            for i=1:K
                Xci = spmb_matmul(Wdc,Xc(:,i)','dim',2);
                Xdi = spmb_matmul(Wcd,Xd(:,i)','dim',2);
                H(:,i,c,i,d) = spmb_dot(Xci,Xd(:,i)',2);
                H(:,i,d,i,c) = H(:,i,c,i,d);
                for j=i+1:K
                    Xcj = Xc(:,j)';
                    Xdj = Xd(:,j)';
                    H(:,i,c,j,d) = spmb_dot(Xci,Xdj,2);
                    H(:,j,c,i,d) = spmb_dot(Xcj,Xdi,2);
                    H(:,i,d,j,c) = H(:,j,c,i,d);
                    H(:,j,d,i,c) = H(:,i,c,j,d);
                end
            end
        end
    end
end
if Cb > 1
    H = spmb_full2sym(reshape(H, [], KC, KC),'dim',2); 
    H = reshape(H, [], K, C, K, C);
end

% =========================================================================
function F = smart_gradient(W,F,wmode)
if wmode == 'D', F = W .* F;
else,            F = spm_squeeze(spmb_sym_rmatmul(spm_unsqueeze(F,2),W,'dim',2),2);
end