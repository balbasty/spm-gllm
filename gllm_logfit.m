function [B,F] = gllm_logfit(Y,X,W,opt)
% Loglinear fit of a loglinear model.
%
% FORMAT [B,F] = gllm_logfit(Y,X,W,opt)
%
% Y - (N x M)       Observations
% X - (M x K)       Design matrix
% W - (N x M x M)   Precision matrix (= WLS weights) [1]
% B - (N x K)       Model parameters
% F - (N x M)       Signal fits
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of parameters
%__________________________________________________________________________
%
% opt.iter - (int)                            Reweighting iterations    [0]
% opt.mode - ('auto'|'full'|'sym'|'estatics') Inversion mode         [auto]
% opt.proc - (handle)                         Postprocessing handle      []
%__________________________________________________________________________
%
% The model of the signal is F = exp(B*X').
%
% This function solves the linear system
%
%                  log(F) = B*X'   =>   B = log(F) / X';
%
% (or its weighted version) with respect to the model parameters B.
%__________________________________________________________________________
%
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

if nargin < 3, W   = 1;      end
if nargin < 4, opt = struct; end

if ~isfield(opt, 'iter'), opt.iter = 0;      end
if ~isfield(opt, 'mode'), opt.mode = 'auto'; end
if ~isfield(opt, 'proc'), opt.proc = @(B) B; end

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
end

% -------------------------------------------------------------------------
% If reweighting, run loop

% The idea behind reweighting is that, in a first order approximation, the
% standard deviation of the log-transformed data scales with the inverse
% of the true signal. After a given fit, we can therefore update our
% estimate of the true signal and run a new WLS fit with proper weights.
%
% Note that the log transform also induces a bias term -0.5*(sigma/mu)^2
% which we do not correct for.
%
% See: https://stats.stackexchange.com/questions/57715/
if opt.iter
    iter     = opt.iter;
    opt.iter = 0;
    [B,F]    = gllm_logfit(Y,X,W,opt);
    for i=1:iter
         A    = smart_FWF(W,F,wmode);
        [B,F] = gllm_logfit(Y,X,A,opt);
    end
    return;
end

% -------------------------------------------------------------------------
% Otherwise, run OLS/WLS

Y = log(max(Y, 1e-12));

% -------------------------------------------------------------------------
% Ordinary Least Squares (OLS)
if isscalar(W)
    B = Y / X';

% -------------------------------------------------------------------------
% Weighted Least Squares (WLS) -- same system across voxels
elseif size(W, 1) == 1
    W  = spm_squeeze(W,1);

    switch wmode
    case 'D', XW = (X .* W)';
    case 'F', XW = (X' * W);
    case 'S', XW = spmb_sym_lmatmul(W,X)';
    end
    % ================================================
    % Solve
    B  = (Y * XW') / (XW * X);

% -------------------------------------------------------------------------
% Weighted Least Squares (WLS) -- different systems across voxels
else
    X  = spm_unsqueeze(X,1);

    switch mode
    % ================================================
    % ESTATICS solver
    case 'E'
        % We know that the precision must be diagonal
        Y             = (Y .* W);
        XX            = zeros([N 2*K-1], class(X));
        XX(:,1:K)     = spmb_dot(W, X.^2, 2, 'squeeze');
        XX(:,K+1:end) = spmb_dot(W, X(:,:,1:end-1).*X(:,:,end), 2, 'squeeze');
        % --------------------------------------------
        % solve with ESTATICS solver
        B = spmb_estatics_solve(XX, Y, 'dim', 2);

    % ================================================
    % Symmetric solver
    case 'S'
        switch wmode
        % --------------------------------------------
        % W Diagonal
        case 'D'
            Y         = (Y .* W);
            XX        = zeros([N (K*(K+1))/2], class(X));
            XX(:,1:K) = spmb_dot(W, X.^2, 2, 'squeeze');
            i = K;
            for j=1:K
                XX(:,i+1:i+K-j) = spmb_dot(W, X(:,:,j+1:end).*X(:,:,j), 2, 'squeeze');
                i = i + K-j;
            end
        % --------------------------------------------
        % W Full
        case 'F'
            Y  = spmb_matmul(W, Y, 'dim', 2);
            W  = spmb_full2sym(W, 'dim', 2);
            XX = spmb_sym_outer(permute(X,[1 3 2]), W, 'dim', 2);
        % --------------------------------------------
        % W Compact
        case 'S'
            Y = spmb_sym_lmatmul(W,Y,'dim',2);
            XX = spmb_sym_outer(permute(X,[1 3 2]), W, 'dim', 2);
        end
        Y  = Y * X;
        % --------------------------------------------
        % solve with spm_field solver
        XX = reshape(XX, 1, 1, N, []);
        Y  = reshape(Y,  1, 1, N, []);
        B  = spm_field(XX, Y, [1 1 1 0 0 0 1 1]);
        B  = reshape(B, N, []);

    % ================================================
    % Generic solver
    otherwise

        switch wmode
        % --------------------------------------------
        % W Diagonal
        case 'D'
            Y  = (Y .* W);
            XX = spm_unsqueeze(X, 3) .* spm_unsqueeze(X, 4);
            XX = spm_dot(W, XX, 2, 'squeeze');
        % --------------------------------------------
        % W Full
        case 'F'
            Y  = spmb_matmul(W, Y, 'dim', 2);
            W  = spmb_full2sym(W, 'dim', 2);
            XX = spmb_sym_outer(permute(X,[1 3 2]), W, 'dim', 2);
            XX = spmb_sym2full(XX, 'dim', 2);
        % --------------------------------------------
        % W Compact
        case 'S'
            Y  = spmb_sym_lmatmul(W,Y,'dim',2);
            XX = spmb_sym_outer(permute(X,[1 3 2]), W, 'dim', 2);
            XX = spmb_sym2full(XX, 'dim', 2);
        end
        Y  = Y * X;
        % --------------------------------------------
        % solve with parfor (slow...)
        B  = zeros([N K], class(Y(1)/X(1)));
        parfor n=1:N
            B(n,:) = Y(n,:) / spm_squeeze(XX(n,:,:),1);
        end

    end
end

% -------------------------------------------------------------------------
% Return
B = opt.proc(B);
if nargout > 1
    X = spm_squeeze(X,1);
    F = exp(B * X');
end

% =========================================================================
function W = smart_FWF(W,F,wmode)
switch wmode
% -------------------------------------
% A: Full matrix
case 'F'
    W = W .* (F .* spm_unsqueeze(F,2)); 
% -------------------------------------
% A: Symmetric matrix
case 'S'
    F = spmb_sym_outer(F,'dim',2);
    W = W .* F;
% -------------------------------------
% A: Diagonal
otherwise
    W = W .* (F.*F);
end
