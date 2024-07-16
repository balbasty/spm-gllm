function [B,F] = gllm_fit(Y,X,W,opt)
% Nonlinear fit of a (sum of) loglinear model(s).
%
% FORMAT [B,F] = gllm_fit(Y,X,W,opt)
%
% Y - (N x M)       Observations
% X - (M x K x C?)  Design matrix
% W - (N x M x M)   Precision matrix (= WLS weights) [1]
% B - (N x K x C?)  Model parameters
% F - (N x M)       Signal fits
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of parameters
% C - Number of components
%__________________________________________________________________________
%
% opt.mc    - (int)                            Number of model comp.    [1]
% opt.iter  - (int)                            Number of iterations    [32]
% opt.tol   - (float)                          Early stop tolerance   [eps]
% opt.mode  - ('auto'|'full'|'sym'|'estatics') Inversion mode        [auto]
% opt.proc  - (handle)                         Postprocessing            []
% opt.init  - ('logfit'|(1|N x K x C?))        Initial estimate    [logfit]
% opt.accel - (float)                          Robustness trade-off     [0]
% opt.verb  - (int)                            Verbosity                [0]
% opt.prior - (N x K x C?)           !TODO!    Prior expected value     [0]
% opt.prec  - (N x K x K x C? x C?)  !TODO!    Prior precision          [0]
%__________________________________________________________________________
%
% Single component model
% ----------------------
% The model of the signal is F = exp(B*X').
%
% This function minimizes the mean squared error
%
%                       0.5 * trace((Y-F)'*(Y-F))
%
% (or its weighted version) with respect to the model parameters B.
%
% If mutiple compartments are provided, the model of the signal is a sum
% of exponentials.
%
% Multi component model
% ---------------------
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
%   number of independent model parameters `opt.mc` is set to `C > 1`, 
%   each component has its own set of parameters, on which act a shared
%   design matrix:
%
%                   F = \sum_c exp(B(:,:,c) * X')
%
%   In this case, the returned parameter matrix B has shape (N x K x C)
%
% * Finally, both cases can be combined, in which case each model in the
%   sum has its own design matrix and set of parameters:
%
%                   F = \sum_c exp(B(:,:c) * X(:,:,c)')
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

USE_SPM_FIELD = true;

% -------------------------------------------------------------------------
% Options
if nargin < 3,             W         = 1;        end
if nargin < 4,             opt       = struct;   end
if ~isfield(opt, 'iter'),  opt.iter  = 64;       end
if ~isfield(opt, 'mode'),  opt.mode  = 'auto';   end
if ~isfield(opt, 'proc'),  opt.proc  = @(B) B;   end
if ~isfield(opt, 'init'),  opt.init  = 'logfit'; end
if ~isfield(opt, 'accel'), opt.accel = 0;        end
if ~isfield(opt, 'verb'),  opt.verb  = 0;        end
if ~isfield(opt, 'mc'),    opt.mc    = 1;        end

% Default tolerance based on data type
type = class(Y(1)*X(1)*W(1)*1.0);
if ~isfield(opt, 'tol')
    opt.tol = eps(type);
end

% -------------------------------------------------------------------------
% Checks sizes
N  = size(Y,1);
M  = size(X,1);
K  = size(X,2);
Cx = size(X,3);
Cb = opt.mc;
C  = max(Cx,Cb);

if size(Y,2) ~= M                       ...
|| ~any(size(W,1) == [1 N])             ...
|| ~any(size(W,2) == [1 M (M*(M+1))/2]) ...
|| ~any(size(W,3) == [1 M])             ...
|| ~any(Cb        == [1 Cx])
    error('Incompatible matrix sizes')
end

% If multi-compartment, replace default `gllm_diff` with `gllm_mc_diff`
if C ~= 1, gllm_diff = @gllm_mc_diff; end

% -------------------------------------------------------------------------
% Set precision and Hessian layout
if ~ismatrix(W),        wmode = 'F';    % Full
elseif size(W,2) > M,   wmode = 'S';    % Compact symmetric
else,                   wmode = 'D';    % Compact diagonal
end

% Autoset Hessian layout
mode = upper(opt.mode(1));
if C > 1,           if ~any(mode == 'FS'), mode = 'S'; end  % Symmetric
elseif mode == 'A', if glm_is_estatics(X), mode = 'E';      % ESTATICS
                    else,                  mode = 'S'; end  % Symmetric
end

% If precision not diagonal -> cannot use ESTATICS layout
if mode == 'E' && wmode ~= 'D',            mode  = 'S'; end

% If precision is full      -> convert to same layout as Hessian
if mode == 'S' && wmode == 'F'
    wmode = 'S';
    W     = spmb_full2sym(W,'dim',2);
end

% -------------------------------------------------------------------------
% Initialize model parameters
B = init_params(Y,X,W,Cb,opt.init,type);
B = opt.proc(B);

% -------------------------------------------------------------------------
% Pre-compute logdet(W)
if opt.verb, switch wmode
case 'F', LW = mean(mean(log(spmb_sym_chol(spmb_full2sym(W,2),2)),2)) * 2;
case 'S', LW = mean(mean(log(spmb_sym_chol(W,2)),2)) * 2;
case 'D', LW = mean(mean(log(W))); end
else,     LW = 0; end

% -------------------------------------------------------------------------
% Optimization loop
opt_diff.mode  = mode;
opt_diff.accel = opt.accel;
msg = '';
L0  = inf;
for i=1:opt.iter

    % ---------------------------------------------------------------------
    % Compute derivatives
    [F,G,H] = gllm_diff(Y,X,B,W,opt_diff);

    G = reshape(G, N, []);
    if     mode == 'F', H = reshape(H, N, K*Cb, K*Cb);
    elseif mode == 'S', H = reshape(H, N, []); end

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
    B = reshape(B, [N K Cb]);
    B = opt.proc(B);
end
if opt.verb, fprintf('\n'); end

% -------------------------------------------------------------------------
% Return
if nargout > 1
    Fc = exp(B(:,:,1)*X(:,:,1)');
    for c=2:C
        Fc = Fc + exp(B(:,:,min(c,Cb))*X(:,:,min(c,Cx))');
    end
end

% =========================================================================
function B = init_params(Y,X,W,Cb,B0,type)
N  = size(Y,1);
K  = size(X,2);
Cx = size(X,3);
C  = max(Cx,Cb);

% -------------------------------------------------------------------------
% Fill with zeros
if isempty(B0)
    B = zeros([N K Cb], type);
    if C == 1 && glm_is_estatics(X), B(:,end) = 1; end
    return
end

% -------------------------------------------------------------------------
% Initialise with a log-linear fit
if ischar(B0) && strcmpi(B0, 'logfit')

    % --------------------------------
    % Log-linear fit
    if C == 1
        B = gllm_logfit(Y,X,W,struct('mode',mode,'proc',opt.proc));
        return
    end

    % --------------------------------
    % Log-linear fit
    % + sample components from posterior
    XX = X;
    if Cb == 1
        XX = sum(X,3);
        if rcond(XX' * XX) == 0
            warning(['Cannot initialise with a logfit: ' ...
                     'design matrix is ill-posed and not identical ' ...
                     'across components. Random initialisation instead.'])
            B = randn([N K Cb], type);
            return;
        end
    end

    for cc=1:size(XX,3)
        X1 = XX{cc};
        B1 = gllm_logfit(Y/C,X1,W,struct('mode',mode,'proc',opt.proc));
        % Compute posterior covariance
        if wmode == 'D'
            S = spm_unsqueeze(X1',1) .* sqrt(spm_unsqueeze(W,2));
            S = spmb_outer(S,'dim',2);
        else
            S = spmb_outer(spm_unsqueeze(X1',1),W,'dim',2);
        end
        % Cholesky decomposition
        S = spmb_chol(spmb_sym_inv(S,'dim',2),'dim',2);
            
        % Sample from multivariate normal
        if Cb==Cx, crange=cc; else, crange=1:Cb; end
        B  = zeros([N K C],class(B1));
        for c=crange
            B(:,:,c) = B1 + spmb_tril_matvec(S,randn([N K]),'dim',2);
        end
    end
    return;
end

% -------------------------------------------------------------------------
% Fill with same value + spread
% (otherwise sum is not separable)
B = zeros([N K Cb], type);
B(:,:,:) = B0;
if size(B0,3) == 1 && Cb > 1
    B = B + randn([N K Cb], type);
end