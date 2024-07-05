function [B,F] = gllm_fit(Y,X,W,opt)
% Nonlinear fit of a loglinear model.
%
% FORMAT [B,F] = loglin_logfit(X,D,W,opt)
%
% Y - (N x M)   Observations
% X - (M x K)   Design matrix
% W - (1|N x M) Weights for WNLLS [1]
% B - (N x K)   Model parameters
% F - (N x M)   Signal fits
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of parameters
%__________________________________________________________________________
%
% opt.iter  - (int)                            Number of iterations [20]
% opt.tol   - (float)                          Early stop tolerance [eps]
% opt.mode  - ('auto'|'full'|'sym'|'estatics') Inversion mode       [auto]
% opt.proc  - (handle)                         Postprocessing       []
% opt.init  - ('logfit'|(1|N x K))             Initial estimate     [logfit]
% opt.accel - (float)                          Robustness trade-off [0]
% opt.verb  - (int)                            Verbosity            [0]
%__________________________________________________________________________
%
% The model of the signal is F = exp(B*X').
%
% This function minimizes the mean squared error
%
%                       0.5 * trace((Y-F)'*(Y-F))
%
% (or its weighted version) with respect to the model parameters B.
%__________________________________________________________________________

% Yael Balbastre

N = size(Y,1);
K = size(X,2);

if nargin < 3, W   = 1;      end
if nargin < 4, opt = struct; end

if ~isfield(opt, 'iter'),  opt.iter  = 20;     end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto'; end
if ~isfield(opt, 'proc'),  opt.proc  = @(B) B; end
if ~isfield(opt, 'accel'), opt.accel = 0;      end
if ~isfield(opt, 'verb'),  opt.verb  = 0;      end

if strcmpi(opt.mode, 'auto')
    if glm_is_estatics(X)
        opt.mode = 'estatics';
    else
        opt.mode = 'sym';
    end
end

% Initialize model parameters
type = class(Y(1)*X(1)*W(1));
if ~isfield(opt, 'init') | isempty(opt.init)
    B = zeros([N K], type);
    B(:,end) = 1;
elseif strcmpi(opt.init, 'logfit')
    B = gllm_logfit(Y,X,W,struct('mode',opt.mode,'proc',opt.proc));
else
    B = zeros([N K], type);
    B(:,:) = opt.init;
end

% Default tolerance based on data type
if ~isfield(opt, 'tol')
    opt.tol = eps(class(B));
end

% Optimization loop
LW = mean(mean(log(W)));
L0 = inf;
opt_diff.mode  = opt.mode;
opt_diff.accel = opt.accel;
for i=1:opt.iter
    % Compute derivatives
    [F,G,H] = gllm_diff(Y,X,B,W,opt_diff);

    % Check for early stopping
    L = 0.5 * (mean(mean(W.*(Y-F).^2)) - LW);
    gain = (L0-L)/L0;
    if opt.verb
        fprintf('%2d | %10.6g | gain = %g', i, L, gain);
        if opt.verb >= 2
            fprintf('\n');
        else
            fprintf('\r');
        end
    end
    if gain < opt.tol
        break
    end
    L0 = L;

    % Newton-Raphson update
    if strcmpi(opt.mode, 'estatics')
        % solve with ESTATICS solver
        B = B - spmb_estatics_solve(H, G, 'dim', 2);

    elseif strcmpi(opt.mode, 'sym')
        % solve with spm_field solver
        H = reshape(H, 1, 1, N, []);
        G = reshape(G,  1, 1, N, []);
        B = B - reshape(spm_field(H, G, [1 1 1 0 0 0 1 1]), N, []);

    else
        % solve with parfor (slow...)
        B = zeros([N K], class(X));
        parfor m=1:N
            B(m,:) = B(m,:) - squeeze(H(m,:,:)) \ squeeze(G(:,m));
        end

    end
    B = opt.proc(B);
end
if opt.verb == 1
    fprintf('\n');
end

if nargout > 1
    F = exp(B * X');
end
