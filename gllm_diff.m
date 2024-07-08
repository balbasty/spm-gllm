function [F,G,H] = gllm_diff(Y,X,B,W,opt)
% Derivatives of a loglinear model.
%
% FORMAT [F,G,H] = gllm_diff(Y,X,B,W,opt)
%
% Y - (N x M)       Observations
% X - (M x K)       Design matrix
% B - (N x K)       Model parameters
% W - (N x M x M)   Precision matrix (= WLS weights) [1]
% F - (N x M)       Signal fits
% G - (N x K)       Gradients
% H - (N x K2)      Hessians
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of parameters
%__________________________________________________________________________
%
% opt.accel - (float)                          Robustness trade-off  [0]
% opt.mode  - ('auto'|'full'|'sym'|'estatics') Hessian mode          [auto]
%__________________________________________________________________________
%
% The model of the signal is F = exp(B*X').
%
% This function differentiates the mean squared error
%
%                       0.5 * trace((Y-F)'*(Y-F))
%
% (or its weighted version) with respect to the model parameters B.
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
% * If mode is 'sym', a compact flattened representation of the Hessians 
%   is returned, with shape [N K*(K+1)/2]. The flattened vector contains
%   the main diagonal of the Hessian followed by each row of its upper
%   triangular part: [H(1,1) H(2,2) H(3,3) H(1,2) H(1,3) H(2,3)]
%
% * If mode is 'estatics', the design matrix D must encode an ESTATICS 
%   model (i.e., multiple time series with shared slopes but different 
%   intercepts) with the first K-1 colums of B encoding the intercepts 
%   and the last column encoding the decay rate. In this case, an even more 
%   compact representation of the Hessian matrix is returned:
%   [H(1,1) H(2,2) H(3,3) H(2,1) H(3,1)]
%
% * If the mode is not provided, it is 'estatics' if an ESTATICS-like  
%   design matrix is detected, and 'sym' otherwise.
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
% Options

if nargin < 4, W   = 1;      end
if nargin < 5, opt = struct; end

if ~isfield(opt, 'accel'), opt.accel = 0;      end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto'; end

mode = upper(opt.mode(1));
if mode == 'A'
    if glm_is_estatics(X), mode = 'E';
    else,                  mode = 'S'; end
end

% -------------------------------------------------------------------------
% Checks sizes

N = size(Y,1);
M = size(X,1);
K = size(X,2);

if size(Y,2) ~= M                       ...
|| size(B,1) ~= N                       ...
|| size(B,2) ~= K                       ...
|| ~any(size(W,1) == [1 N])             ...
|| ~any(size(W,2) == [1 M (M*(M+1))/2]) ...
|| ~any(size(W,3) == [1 M])
    error('Incompatible matrix sizes')
end

if ~ismatrix(W),        wmode = 'F';    % Full
elseif size(W,2) > M,   wmode = 'S';    % Compact symmetric
else,                   wmode = 'D';    % Compact diagonal
end

if mode == 'E' && wmode ~= 'D'
    mode = 'S';
end
if mode == 'S' && wmode == 'F'
    W = spmb_full2sym(W,'dim',2);
    wmode = 'S';
end

% -------------------------------------------------------------------------
% Fitted signal
F = exp(B*X');

if nargout == 1, return; end

% -------------------------------------------------------------------------
% Residuals
R = F - Y;

% -------------------------------------------------------------------------
% Gradient
FR = smart_WF(W, F .* R, wmode);
G  = FR * X;

if nargout == 2, return; end

% -------------------------------------------------------------------------
% Hessian
FF = smart_FWF(W, F, wmode);
if opt.accel ~= 1
    if wmode == 'F'
        AFR       = abs(FR);
        FR        = FF;
        idx       = spm_diagind(M);
        FR(:,idx) = FR(:,idx) + AFR * (1 - opt.accel);
    else
        AFR       = abs(FR);
        FR        = FF;
        FR(:,1:M) = FR(:,1:M) + AFR * (1 - opt.accel);
    end
else
    FR = FF;
end

switch mode
% ------------------------------
% H: ESTATICS storage
case 'E'
    H = zeros([N, 2*K-1], class(B));
    H(:,1:K)     = FR * (X.*X);
    H(:,K+1:end) = FF * (X(:,1:end-1).*X(:,end));

% ------------------------------
% H: Compact storage
case 'S'
    switch wmode
    case 'D'
        H = zeros([N, (K*(K+1))/2], class(B));
        H(:,1:K) = FR * (X.*X);
        i = K;
        for j=1:K
            H(:,i+1:i+K-j) = FF * (X(:,j+1:end).*X(:,j));
            i = i + K-j;
        end
    case 'S'
        H = spmb_sym_outer(spm_unsqueeze(X',1), FF, 'dim', 2);
    end

% ------------------------------
% H: Full matrix
otherwise
    XX = spm_unsqueeze(X, -1) .* spm_unsqueeze(X, -2);
    XX = reshape(XX, [], K*K);
    H  = reshape(FF * XX, [], K, K);
    if opt.accel ~= 1
        msk = spm_unsqueeze(eye(K, 'logical'),1);
        H = H .* ~msk + reshape(FR * XX, [], K, K) .* msk;
    end
end

% =========================================================================
function W = smart_FWF(W,F,wmode)
switch wmode
case 'F', W = W .* (F .* spm_unsqueeze(F,2)); 
case 'S', W = W .* spmb_sym_outer(F,'dim',2);
case 'D', W = W .* (F.*F);
end

% =========================================================================
function F = smart_WF(W,F,wmode)
switch wmode
case 'F'
    sqz = size(W,1) == 1;
    if sqz, F = F * spm_squeeze(W,1);
    else,   F = spmb_matmul(F,W,'dim',2); end
case 'S',   F = spm_squeeze(spmb_sym_rmatmul(spm_unsqueeze(F,2),W,'dim',2),2);
case 'D',   F = W .* F;
end