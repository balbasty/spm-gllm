function [B,F] = gllm_logfit(Y,X,W,opt)
% Loglinear fit of a loglinear model.
%
% FORMAT [B,F] = loglin_logfit(X,D,W,opt)
%
% Y - (N x M)   Observations
% X - (M x K)   Design matrix
% W - (1|M x N) Weights for WLS [1]
% B - (N x K)   Model parameters
% F - (N x M)   Signal fits
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

% Yael Balbastre

N = size(Y,1);
K = size(X,2);

if nargin < 3, W   = 1;      end
if nargin < 4, opt = struct; end

if ~isfield(opt, 'iter'), opt.iter = 0;      end
if ~isfield(opt, 'mode'), opt.mode = 'auto'; end
if ~isfield(opt, 'proc'), opt.proc = @(B) B; end

if strcmpi(opt.mode, 'auto')
    if glm_is_estatics(X)
        opt.mode = 'estatics';
    else
        opt.mode = 'sym';
    end
end

% If reweighting, run loop
%
% The idea behind reweighting is that, in a first order approximation, the
% standard deviation of the log-transformed data scales with the inverse
% of the true signal. After a given fit, we can therefore update our
% estimate of the true signal and run a new WLS fit with proper weights.
if opt.iter
    iter = opt.iter;
    opt.iter = 0;
    [B,F] = gllm_logfit(Y,X,W,opt);
    for i=1:iter
        [B,F] = gllm_logfit(Y,X,W.*(F.*F),opt);
    end
    return;
end

% Otherwise, run OLS/WLS

Y = log(max(Y, 1e-12));

if all(W == 1)
    % Ordinary Least Squares (OLS)
    B = Y / X';

elseif size(W, 1) == 1
    % Weighted Least Squares (WLS) -- same system across voxels
    Y  = (Y  .* W) * X;
    XX = (X' .* W) * X;
    B  = Y / XX;

else
    % Weighted Least Squares (WLS) -- different systems across voxels
    Y  = (Y .* W) * X;

    X  = spm_unsqueeze(X,1);
    if strcmpi(opt.mode, 'estatics')
        XX            = zeros([N 2*K-1], class(X));
        XX(:,1:K)     = spmb_dot(W, X.^2, 2, 'squeeze');
        XX(:,K+1:end) = spmb_dot(W, X(:,:,1:end-1).*X(:,:,end), 2, 'squeeze');

        % solve with ESTATICS solver
        B = spmb_estatics_solve(XX, Y, 'dim', 2);

    elseif strcmpi(opt.mode, 'sym')
        XX            = zeros([N (K*(K+1))/2], class(X));
        XX(:,1:K)     = spmb_dot(W, X.^2, 2, 'squeeze');
        i = K;
        for j=1:K
            XX(:,i+1:i+K-j) = spmb_dot(W, X(:,:,j+1:end).*X(:,:,j), 2, 'squeeze');
            i = i + K-j;
        end

        % solve with spm_field solver
        XX = reshape(XX, 1, 1, N, []);
        Y  = reshape(Y,  1, 1, N, []);
        B  = spm_field(XX, Y, [1 1 1 0 0 0 1 1]);
        B  = reshape(B, N, []);

    else
        XX = spm_unsqueeze(X, 3) .* spm_unsqueeze(X, 4);
        XX = spm_dot(W, XX, 2, 'squeeze');

        % solve with parfor (slow...)
        B  = zeros([N K], class(Y(1)/X(1)));
        parfor n=1:N
            B(n,:) = XX(n,:,:) \ Y(n,:);
        end

    end
end

B = opt.proc(B);
if nargout > 1
    X = spm_squeeze(X,1);
    F = exp(B * X');
end
