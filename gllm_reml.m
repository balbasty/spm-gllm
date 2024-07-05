function [C,B,h,P] = gllm_reml(Y,X,Q,opt)
% ReML estimate of covariance in a log-linear model
%
% FORMAT [C,B,h,P] = simple_reml(Y,X,Q,opt)
%__________________________________________________________________________
%
% Y  - (N x M)     Observed data
% X  - (M x K)     Design matrix
% Q  - (J x M)     Covariance basis set (diagonals)
% C  - (M x 1)     Estimated diagonal covariance: C = sum(h*Q,1)
% B  - (N x K)     Expected model parameters
% h  - (J x 1)     Parameters of the covariance in the basis set
% P  - (J x J)     Posterior precision of h
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of model parameters
% J - Number of bases in the covariance basis set
%__________________________________________________________________________
% 
% opt.N     - Number of voxels                                          [1]
% opt.lam   - Incomplete integral regularization (1/t)               [0.25]
% opt.lev   - Levenberg regularization                                  [0]
% opt.mqd   - Marquardt regularization                               [1e-8]
% opt.accel - Hessian weight (robust:0...1:fisher)                    [0.5]
% opt.iter  - Maximum number of EM iterations                          [32]
% opt.tol   - Tolerance for early stopping                              [0]
% opt.hE    - Hyperprior mean                                           [0]
% opt.hP    - Hyperprior precision                                [exp(-8)]
% opt.verb  - Verbosity                                                 [0]
% opt.fit   - Options passed to loglin_fit                               []
%__________________________________________________________________________

% Yael Balbastre

if nargin < 3,             Q         = [];      end
if nargin < 4,             opt       = struct;  end
if ~isfield(opt, 'N'),     opt.N     = 1;       end
if ~isfield(opt, 'lam'),   opt.lam   = 1/4;     end
if ~isfield(opt, 'lev'),   opt.lev   = 0;       end
if ~isfield(opt, 'mqd'),   opt.mqd   = 0;       end
if ~isfield(opt, 'accel'), opt.accel = 0;     end
if ~isfield(opt, 'iter'),  opt.iter  = 128;     end
if ~isfield(opt, 'tol'),   opt.tol   = 0;       end
if ~isfield(opt, 'hE'),    opt.hE    = 0;       end
if ~isfield(opt, 'hP'),    opt.hP    = exp(-8); end
if ~isfield(opt, 'verb'),  opt.verb  = 0;       end
if ~isfield(opt, 'fit'),   opt.fit   = struct;  end

N = size(Y,1);
M = size(Y,2);
if isempty(Q)
    Q = zeros([M M]);
    for j=1:M
        Q(j,j) = 1;
    end
end
J = size(Q,1);

% Initial fit + homoscedastic variance
B = gllm_fit(Y,X,1,opt.fit);
R = exp(B*X') - Y;
h = dot(R(:),R(:)) / numel(R);

% Initialize hyperparameter
h = h * ones([J 1], class(X));

% Set hyperprior
hE = zeros([J 1]);
hE(:) = opt.hE(:);
if isscalar(opt.hP)
    hP = zeros([J J]);
    hP(:) = opt.hP;
elseif isvector(opt.hP)
    hP = diag(opt.hP);
else
    hP = zeros([J J]);
    hP(:,:) = opt.hP(:,:);
end

for iter=1:opt.iter
    % Compute precision = WNLS weights
    C = sum(h .* Q, 1);              % Current variance
    A = 1./C(:);                     % Inverse variance (= precision)

    % Run WNLS fit
    opt.fit.init = B;
    B = gllm_fit(Y,X,A',opt.fit);
    R = residual_matrix(Y,X,B,A) / N;
   
    % Utility matrices
    P = diag(A) - (A .* R) .* A';
    A = spm_unsqueeze(A,1);
    P = spm_unsqueeze(P,1);

    % Hessian loading
    HH = 2 * spm_squeeze(A,1) .* R - eye(M);
    HH = sqrt(sum(abs(HH)));
    HH = HH.^(1-opt.accel);

    % Likelihood
    g = spmb_trace(P .* Q, 'dim', 2);
    H = spmb_sym2full(spmb_sym_outer(A .* Q .* HH));

    % Prior
    g = g + hP * (h - hE);
    H = H + hP;

    % Loading
    if opt.lev
        H = H + opt.lev * eye(J);
    end
    if opt.mqd
        H = H + opt.mqd * max(diag(H)) * eye(J);
    end

    % Solve system
    if opt.lam
        dh = spm_dx(-H,g,{1/opt.lam});
    else
        dh = H\g;
    end
    h = h - dh;

    % Loss
    A    = spm_squeeze(A,1);
    loss = trace(A.*R) - sum(log(A));
    ghg  = g'*dh;
    if opt.verb
        fprintf('%2d | %10.6g | %10.6g', iter, loss/M, ghg/M);
        if opt.verb >= 2
            fprintf('\n');
        else
            fprintf('\r');
        end
    end
    if ghg < opt.tol, break; end
end
if opt.verb == 1, fprintf('\n'); end

P = H;
end

function R = residual_matrix(Y,X,B,A)
M  = size(Y,2);
K  = size(X,2);
Z  = exp(B*X');
S  = spmb_sym_outer(Z, 'dim', 2);
S(:,1:M) = S(:,1:M) .* spm_unsqueeze(A,1);
S  = spmb_sym_outer(spm_unsqueeze(X',1), S, 'dim', 2); % Precision about B
S(:,1:K) = S(:,1:K) + max(abs(S(:,1:K)),[],2) * 1e-8;
S  = spmb_sym_inv(S, 'dim', 2);                        % Uncertainty about B
S  = spmb_sym_outer(spm_unsqueeze(X,1), S, 'dim', 2);  % Uncertainty about BX'
S  = exp(S);
ZZ = spmb_sym_outer(Z,'dim',2);                        % \hat{z} * \hat{z}'
ZZ = spm_squeeze(dot(ZZ, S, 1), 1);                    % \sum_n E[z(n) * z(n)']
ZZ = spmb_sym2full(ZZ);
YZ = Z .* sqrt(S(:,1:M));                              % E[z(n)]
YZ = spmb_dot(spm_unsqueeze(YZ,2), Y, 1,'squeeze');    % \sum_n E[z(n) * y(n)']
YZ = YZ + YZ';
YY = Y' * Y;
R  = ZZ + YY - YZ;
end