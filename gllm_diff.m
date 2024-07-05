function [F,G,H] = gllm_diff(Y,X,B,W,opt)
% Derivatives of a loglinear model.
%
% FORMAT [F,G,H] = loglin_diff(Y,X,B,W,opt)
%
% Y - (N x M)   Observations
% X - (M x K)   Design matrix
% B - (N x K)   Model parameters
% W - (1|M x N) Obervation weights [1]
% F - (M x N)   Signal fits
% G - (M x K)   Gradients
% H - (M x K2)  Hessians
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
%__________________________________________________________________________

% Yael Balbastre

N = size(Y,1);
K = size(X,2);

if nargin < 4, W   = 1;      end
if nargin < 5, opt = struct; end

if ~isfield(opt, 'accel'), opt.accel = 0;      end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto'; end

if strcmpi(opt.mode, 'auto')
    if is_estatics(X)
        opt.mode = 'estatics';
    else
        opt.mode = 'sym';
    end
end

% Fitted signal
F = exp(B*X');

if nargout > 1

    % Residuals
    R = F - Y;
    
    % Gradient
    FR = W .* (F .* R);
    G  = FR * X;
    
    if nargout > 2
        % Hessian
        FF = W .* (F .* F);
        FR = FF + abs(FR) * (1 - opt.accel);

        if strcmpi(opt.mode, 'estatics')
            H = zeros([N, 2*K-1], class(B));
            H(:,1:K)     = FR * (X.*X);
            H(:,K+1:end) = FF * (X(:,1:end-1).*X(:,end));

        elseif strcmpi(opt.mode, 'sym')
            H = zeros([N, (K*(K+1))/2], class(B));
            H(:,1:K) = FR * (X.*X);
            i = K;
            for j=1:K
                H(:,i+1:i+K-j) = FF * (X.*X(:,j));
                i = i + K-j;
            end

        else
            XX = unsqueeze(X, -1) .* unsqueeze(X, -2);
            XX = reshape(XX, [], K*K);
            H  = reshape(FF * XX, [], K, K);
            if accel ~= 1
                msk = eye(K, 'logical');
                H = H .* ~msk + reshape(FR * XX, [], K, K) * msk;
            end
        end
    end

end

end