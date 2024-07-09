function [B,F] = gllm_fit(Y,X,W,opt)
% Nonlinear fit of a loglinear model.
%
% FORMAT [B,F] = gllm_fit(X,D,W,opt)
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
% opt.iter  - (int)                            Number of iterations    [20]
% opt.tol   - (float)                          Early stop tolerance   [eps]
% opt.mode  - ('auto'|'full'|'sym'|'estatics') Inversion mode        [auto]
% opt.proc  - (handle)                         Postprocessing            []
% opt.init  - ('logfit'|(1|N x K))             Initial estimate    [logfit]
% opt.accel - (float)                          Robustness trade-off     [0]
% opt.verb  - (int)                            Verbosity                [0]
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

USE_SPM_FIELD = true;

% -------------------------------------------------------------------------
% Options

if nargin < 3, W   = 1;      end
if nargin < 4, opt = struct; end

if ~isfield(opt, 'iter'),  opt.iter  = 20;     end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto'; end
if ~isfield(opt, 'proc'),  opt.proc  = @(B) B; end
if ~isfield(opt, 'accel'), opt.accel = 0;      end
if ~isfield(opt, 'verb'),  opt.verb  = 0;      end

% Default tolerance based on data type
type = class(Y(1)*X(1)*W(1)*1.0);
if ~isfield(opt, 'tol')
    opt.tol = eps(type);
end

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
    wmode = 'S';
end

% -------------------------------------------------------------------------
% Initialize model parameters
if ~isfield(opt, 'init') || isempty(opt.init)
    B = zeros([N K], type);
    B(:,end) = 1;
elseif strcmpi(opt.init, 'logfit')
    B = gllm_logfit(Y,X,W,struct('mode',mode,'proc',opt.proc));
else
    B = zeros([N K], type);
    B(:,:) = opt.init;
end

% -------------------------------------------------------------------------
% Optimization loop
if opt.verb, switch wmode
case 'F', LW = mean(mean(log(spmb_sym_chol(spmb_full2sym(W,2),2)),2)) * 2;
case 'S', LW = mean(mean(log(spmb_sym_chol(W,2)),2)) * 2;
case 'D', LW = mean(mean(log(W))); end
else,     LW = 0; end
L0 = inf;
opt_diff.mode  = mode;
opt_diff.accel = opt.accel;
msg = '';
for i=1:opt.iter

    % ---------------------------------------------------------------------
    % Compute derivatives
    [F,G,H] = gllm_diff(Y,X,B,W,opt_diff);

    % ---------------------------------------------------------------------
    % Check for early stopping
    switch wmode
    case 'D', L = mean(mean(W.*(Y-F).^2));
    case 'S', L = mean(spmb_sym_inner(Y-F, W, 'dim', 2)) / M;
    case 'F', L = mean(mean((Y-F) .* spmb_matvec(W,Y-F,2)));
    end
    L = 0.5 * (L - LW);
    gain = (L0-L)/L0;
    if opt.verb
        if opt.verb >= 2, fprintf('\n');
        else,             fprintf(repmat('\b',1,length(msg))); end
        msg = sprintf('%2d | %10.6g | gain = %g', i, L, gain);
        fprintf(msg);
    end
    if gain < opt.tol
        break
    end
    L0 = L;

    switch mode
    % ---------------------------------------------------------------------
    % Newton-Raphson with ESTATICS solver
    case 'E'
        B = B - spmb_estatics_solve(H, G, 'dim', 2);

    % ---------------------------------------------------------------------
    % Newton-Raphson with symmetric solver
    case 'S'
        if USE_SPM_FIELD
            % solve with spm_field solver
            T = class(B);
            H = single(reshape(H, 1, 1, N, []));            
            G = single(reshape(G, 1, 1, N, []));
            B = B - cast(reshape(spm_field(H, G, [1 1 1 0 0 0 1 1]), N, []),T);
        else
            % slower version that uses a batched cholesky decomposition
            B = B - spmb_sym_solve(H, G, 'dim', 2);
        end

    % ---------------------------------------------------------------------
    % Newton-Raphson with generic solver
    otherwise
        B = zeros([N K], class(X));
        parfor m=1:N
            B(m,:) = B(m,:) - G(m,:) / spm_squeeze(H(m,:,:),1);
        end

    end

    % ---------------------------------------------------------------------
    % Eventual postprocessing (value clipping?)
    B = opt.proc(B);
end
if opt.verb, fprintf('\n'); end

% -------------------------------------------------------------------------
% Return
if nargout > 1
    F = exp(B * X');
end
