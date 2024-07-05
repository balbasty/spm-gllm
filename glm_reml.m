function [C,h,P] = glm_reml(YY,X,Q,opt)
% ReML estimate of covariance in a linear model (reimplements spm_reml)
%
% FORMAT [C,h,P] = simple_reml(YY,X,Q,opt)
%__________________________________________________________________________
%
% YY - (M x M)     Observed second moments (/N, spatially whitened)
% X  - (M x K)     Design matrix
% Q  - (J x M x M) Covariance basis set
% C  - (M x M)     Estimated covariance: C = sum(h*Q,1)
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
% opt.N    - Number of voxels                                           [1]
% opt.lam  - Incomplete integral regularization (1/t)                [0.25]
% opt.lev  - Levenberg regularization                                   [0]
% opt.mqd  - Marquardt regularization                                   [0]
% opt.iter - Maximum number of EM iterations                           [32]
% opt.tol  - Tolerance for early stopping                               [0]
% opt.hE   - Hyperprior mean                                            [0]
% opt.hP   - Hyperprior precision                                 [exp(-8)]
% opt.verb - Verbosity                                                  [0]
%__________________________________________________________________________

% Yael Balbastre

if nargin < 4,              opt       = struct;  end
if ~hasfield(opt, 'N'),     opt.N     = 1;       end
if ~hasfield(opt, 'lam'),   opt.lam   = 1/4;     end
if ~hasfield(opt, 'lev'),   opt.lev   = 0;       end
if ~hasfield(opt, 'mqd'),   opt.mqd   = 0;       end
if ~hasfield(opt, 'iter'),  opt.iter  = 32;      end
if ~hasfield(opt, 'tol'),   opt.tol   = 0;       end
if ~hasfield(opt, 'hE'),    opt.hE    = 0;       end
if ~hasfield(opt, 'hP'),    opt.hP    = exp(-8); end

M = size(YY,1);
if nargin < 3
    Q = zeros([M M M]);
    for j=1:M
        Q(j,j,j) = 1;
    end
end
J = size(Q,1);

% Initialize hyperparameter
h = zeros([J 1], class(X));
for i = 1:M
    h(i) = any(diag(squeeze(Q(i,:,:))));
end

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
    C  = sum(h .* Q, 1);    % Current covariance matrix
    A  = inv(C);            % Inverse covariance (= precision) matrix
    AX = A * X;
    BS = inv(X' * AX);      % Posterior covariance of the model parameters

    P  = AX * BS * AX' - A;
    U  = P * YY + eye(M);

    P  = spm_unsqueeze(P,1);
    U  = spm_unsqueeze(U,1);
    PQ = spmb_matmul(P,Q,'dim',2);

    % Likelihood
    g = spmb_trace(PQ,U,'dim',2);
    H = spmb_trace(reshape(PQ,[1 J M M]),reshape(PQ,[J 1 M M]),'dim',3);

    % Prior
    g = g + hP * (h - hE);
    H = H + hP;

    % Loading
    if opt.lev
        H = H + opt.lev * eye(J);
    end
    if opt.mqd
        m = max(diag(H), [], 2);
        H = H + opt.mqd * m * eye(J);
    end

    % Solve system
    if opt.lam
        dh = -spm_dx(-H,g,{1/opt.lam});
    else
        dh = H\g;
    end
    h = h - dh;

    % Loss
    loss = g'*dh;
    if opt.verb
        fprintf('%2d | %10.6g', iter, loss);
        if opt.verb >= 2
            fprintf('\n');
        else
            fprintf('\r');
        end
    end
    if loss < opt.tol, break; end
end
if opt.verb == 1, fprintf('\n'); end

P = H;
end