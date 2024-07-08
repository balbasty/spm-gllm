function [C,B,h,P] = gllm_reml(Y,X,Q,opt)
% ReML estimate of covariance in a log-linear model
%
% FORMAT [C,B,h,P] = gllm_reml(Y,X,Q,opt)
%__________________________________________________________________________
%
% Y - (N x M)     Observed data
% X - (M x K)     Design matrix
% Q - (J x M x M) Covariance basis set [default: diagonal elements]
% C - (M x M)     Estimated covariance: C = sum(h*Q,1)
% B - (N x K)     Expected model parameters
% h - (J x 1)     Parameters of the covariance in the basis set
% P - (J x J)     Posterior precision of h
%__________________________________________________________________________
%
% N - Number of voxels
% M - Number of volumes
% K - Number of model parameters
% J - Number of bases in the covariance basis set
%__________________________________________________________________________
% 
% opt.lam   - Incomplete integral regularization (1/t)               [0.25]
% opt.lev   - Levenberg regularization                                  [0]
% opt.mqd   - Marquardt regularization                                  [0]
% opt.accel - Hessian weight (robust:0...1:fisher)                    [0.5]
% opt.iter  - Maximum number of EM iterations                          [32]
% opt.tol   - Tolerance for early stopping                              [0]
% opt.hE    - Hyperprior mean                                           [0]
% opt.hP    - Hyperprior precision                                [exp(-8)]
% opt.verb  - Verbosity                                                 [0]
% opt.fit   - Options passed to gllm_fit                                 []
%__________________________________________________________________________
%
% The covariance basis Q (and therefore also C) can have shape:
%
%   (J x M x M)     Full    symmetric
%   (J x M2)        Compact symmetric
%   (J x M)         Compact diagonal
%
% where M2 = (M*(M+1))/2
%__________________________________________________________________________

% Yael Balbastre

% -------------------------------------------------------------------------
% Options
if nargin < 3,             Q         = [];      end
if nargin < 4,             opt       = struct;  end
if ~isfield(opt, 'lam'),   opt.lam   = 1/4;     end
if ~isfield(opt, 'lev'),   opt.lev   = 0;       end
if ~isfield(opt, 'mqd'),   opt.mqd   = 0;       end
if ~isfield(opt, 'accel'), opt.accel = 0;       end
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


% -------------------------------------------------------------------------
% Checks sizes

if size(X,1) ~= M                       ...
|| ~any(size(Q,2) == [1 M (M*(M+1))/2]) ...
|| ~any(size(Q,3) == [1 M])
    error('Incompatible matrix sizes')
end

% Layout of the covariance bases
if ~ismatrix(Q),        layout = 'F';    % Full
elseif size(Q,2) > M,   layout = 'S';    % Compact symmetric
else,                   layout = 'D';    % Compact diagonal
end

% If full, make symmetric (simplifies later steps)
Q0 = Q;
QS = Q;
QF = Q;
if layout == 'F'
    QS     = spmb_full2sym(Q,2);
    layout = 'S';
elseif layout == 'S'
    QF     = spmb_sym2full(Q,2);
end

% -------------------------------------------------------------------------
% Initial fit + homoscedastic variance
B = gllm_fit(Y,X,1,opt.fit);
R = exp(B*X') - Y;
h = dot(R(:),R(:)) / numel(R);

% Initialize hyperparameter (minimize KL between C and s^2 * I)
h = init_cov(QS,h,layout);

% Adap mode for weighted gllm_fit
if layout == 'D', opt.fit.mode = 'E';    % ESTATICS
else,             opt.fit.mode = 'S';    % Compact symmetric 
end

% -------------------------------------------------------------------------
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

if opt.verb >= 2
    fprintf([repmat('-', 1, 52) '\n']);
    fprintf('%10s | %10s | %10s | %11s |\n',...
            'it','   loss   ','   g''Hg   ','   time    ');
    fprintf([repmat('-', 1, 52) '\n']);
end

% -------------------------------------------------------------------------
% EM loop
for iter=1:opt.iter
    tic;

    
    % ---------------------------------------------------------------------
    % Compute precision = WNLS weights
    C = spm_squeeze(sum(h .* QS, 1), 1);
    if layout == 'D',   A = 1./C(:);
    else,               A = spmb_sym_inv(C); end
    % ---------------------------------------------------------------------
    % Run WNLS fit
    opt.fit.init = B;
    B = gllm_fit(Y,X,spm_unsqueeze(A,1),opt.fit);
    R = residual_matrix(Y,X,B,A) / N;
   
    % ---------------------------------------------------------------------
    % Utility matrices
    if layout == 'D',   P = diag(A) - (A .* R) .* A';
                        L = 2 * (A .* R) - eye(M);
    else,               A = spmb_sym2full(A);
                        P = A - (A * R) * A;
                        L = 2 * (A * R) - eye(M);
    end

    % ---------------------------------------------------------------------
    % Hessian loading
    L = sqrt(sum(abs(L)));
    L = L.^(1-opt.accel);

    % ---------------------------------------------------------------------
    % Likelihood
    A = spm_unsqueeze(A,1);
    P = spm_unsqueeze(P,1);
    if layout == 'D',   g = spmb_trace(P .* Q, 'dim', 2);
                        H = A .* Q .* L;
                        H = spmb_sym2full(spmb_sym_outer(H));
    else,               g = spmb_trace(P, QF, 'dim', 2);
                        H = spmb_matmul(A, QF, 2);
                        H = spmb_trace(...
                            spm_unsqueeze(H .* L, 1), ...
                            spm_unsqueeze(H.* spm_unsqueeze(L,2), 2), ...
                            'dim', 3);
    end

    % ---------------------------------------------------------------------
    % Prior
    g = g + hP * (h - hE);
    H = H + hP;

    % ---------------------------------------------------------------------
    % Levenberg-Marquardt regularization
    if opt.lev, H = H + opt.lev * eye(J); end
    if opt.mqd, H = H + opt.mqd * max(diag(H)) * eye(J); end

    % ---------------------------------------------------------------------
    % Solve system
    if opt.lam, dh = cast(spm_dx(-double(H),double(g),{1/opt.lam}),class(h));
    else,       dh = H\g; end

    h = h - dh;

    % ---------------------------------------------------------------------
    % Loss
    A    = spm_squeeze(A,1);
    if layout == 'D', loss = trace(A.*R) - sum(log(A));
    else,             loss = trace(A*R)  - spm_logdet(A); end
    ghg  = g'*dh;
    if opt.verb
        fprintf('(reml) %3d | %10.4g | %10.4g | %10.4gs |', iter, loss/M, ghg/M, toc);
        if opt.verb >= 2
            fprintf('\n');
        else
            fprintf('\r');
        end
    end
    if ghg < opt.tol, break; end
end
if opt.verb == 1, fprintf('\n'); end


% Reconstruct covariance
C = sum(h .* Q0, 1);
% Assign posterior precision to var P
P = H;
end

% =========================================================================
function R = residual_matrix(Y,X,B,A)
M  = size(Y,2);
K  = size(X,2);
Z  = exp(B*X');
S  = spmb_sym_outer(Z, 'dim', 2);
A  = spm_unsqueeze(A,1);
if size(A,2) == M
    S(:,1:M) = S(:,1:M) .* A;
else
    S = S .* A;
end
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

% =========================================================================
% Intialize hyper-parameters by minimizing the KL divergence between
% the covariance sum(h.*Q,1) and the homoscedastic variance sigma2*eye
function h = init_cov(Q,sigma2,layout)

if layout == 'S', Q = spmb_sym2full(Q,'dim',2); end

J   = size(Q,1);
M   = size(Q,2);
lam = 1./sigma2;

% -------------------------------------------------------------------------
% Intialize h
if layout == 'F', D = spmb_diag(Q,'dim',2);
else,             D = Q; end
h = zeros(J,1);
for j=1:J
    if any(D(j,:))
        h(j) = sigma2;
    end
end

% -------------------------------------------------------------------------
% Initial state
C = spm_squeeze(sum(h.*Q,1),1) * lam;
if layout == 'D', L = sum(C)   - sum(log(C))   - size(C,1);
else,             L = trace(C) - spm_logdet(C) - size(C,1); end
fprintf('(dkl) %4d | %10.4g |\n', 0, L);

% -------------------------------------------------------------------------
% Loop
armijo = 0.01;
for n=1:32

    % ---------------------------------------------------------------------
    % Gradient and Hessian
    switch layout
    case 'D'
        A = spm_unsqueeze(1./C,1);
        G = sum((1 - A) .* Q, 2);
        H = Q .* A;
        H = spmb_dot(spm_unsqueeze(H,1), spm_unsqueeze(H,2), 3);
    otherwise
        A  = inv(C);
        G  = spm_unsqueeze(eye(M) - A,1);
        G  = spmb_trace(G, Q, 'dim', 2);
        AQ = spmb_matmul(spm_unsqueeze(A,1),Q,'dim', 2);
        H  = zeros(J,class(G));
        for i=1:J
            H(i,i) = trace(spm_squeeze(AQ(i,:,:),1) * ...
                           spm_squeeze(AQ(i,:,:),1) );
            for j=j+1:J
                H(i,j) = trace(spm_squeeze(AQ(i,:,:),1) * ...
                               spm_squeeze(AQ(j,:,:),1) );
                H(j,i) = H(i,j);
            end
        end
    end

    % ---------------------------------------------------------------------
    % Levenberg-Marquardt
    H = H + eye(J) * max(abs(diag(H))) * 1e-8;

    % ---------------------------------------------------------------------
    % Newton step
    H   = H / M;
    G   = G / M;
    dh  = (H\G);
    ghg = (G'*dh)/(M*J);

    % ---------------------------------------------------------------------
    % Line search
    L0 = L;
    armijo = armijo * 1.1;
    dh     = dh * armijo;
    while true
        C = spm_squeeze(sum((h-dh).*Q,1),1) * lam;
        if layout == 'D', L = sum(C)   - sum(log(C))   - size(C,1);
        else,             L = trace(C) - spm_logdet(C) - size(C,1);
        end
        if 0 <= L && L < L0, break; end
        dh     = 0.5 * dh;
        armijo = 0.5 * armijo;
    end
    h  = h - dh; 

    fprintf('(dkl) %4d | %10.4g | %10.4g |\n', n, L, ghg);
    if ghg < 1e-8;  break; end 
end

end