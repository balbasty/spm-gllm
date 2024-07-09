function [C,B,h,P] = gllm_reml(Y,X,Q,opt)
% ReML estimate of covariance in a log-linear model
%
% FORMAT [C,B,h,P] = gllm_reml(Y,X,Q,opt)
%__________________________________________________________________________
%
% Y - (N... x M)  Observed data 
% X - (M x K)     Design matrix
% Q - (J x M x M) Covariance basis set [default: diagonal elements]
% C - (M x M)     Estimated covariance: C = sum(h*Q,1)
% B - (N... x K)  Expected model parameters
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
% opt.accel - Hessian weight (robust:0...1:fisher)                      [0]
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
% USE_SYM uses compact representations for batches of symmetric matrices
% It should use slighly less memory but is much slower.
USE_SYM = false;
if USE_SYM, residual_matrix = @residual_matrix_sym;
else,       residual_matrix = @residual_matrix_full;  end

% -------------------------------------------------------------------------
% Options
if nargin < 3,             Q         = [];      end
if nargin < 4,             opt       = struct;  end
if ~isfield(opt, 'lam'),   opt.lam   = 1/4;     end
if ~isfield(opt, 'lev'),   opt.lev   = 0;       end
if ~isfield(opt, 'mqd'),   opt.mqd   = 0;       end
if ~isfield(opt, 'accel'), opt.accel = 0;       end
if ~isfield(opt, 'iter'),  opt.iter  = 128;     end
if ~isfield(opt, 'tol'),   opt.tol   = eps;       end
if ~isfield(opt, 'hE'),    opt.hE    = 0;       end
if ~isfield(opt, 'hP'),    opt.hP    = exp(-8); end
if ~isfield(opt, 'verb'),  opt.verb  = 0;       end
if ~isfield(opt, 'fit'),   opt.fit   = struct;  end

M = size(X,1);
K = size(X,2);

if isempty(Q)
    Q = zeros([M M]);
    for j=1:M
        Q(j,j) = 1;
    end
end
J = size(Q,1);


% -------------------------------------------------------------------------
% Checks sizes

Yshape = size(Y);
if M == 1, Nd = ndims(Y); if Nd == 2 && size(Y,2) == 1, Nd = 1; end
else,      Nd = ndims(Y)-1; end
if Nd > 1
    Nz = size(Y,Nd);
else
    Nz = 1;
    M  = size(Y,2);
end

if size(Y,Nd+1) ~= M                       ...
|| ~any(size(Q,2) == [1 M (M*(M+1))/2]) ...
|| ~any(size(Q,3) == [1 M])
    error('Incompatible matrix sizes')
end

% Build a cell of handles that load slices on the fly
Y0 = Y;
if Nd > 1, Y = cellfun(@(n) (@() reshape(spm_subsref(Y0,n,Nd), [], M)), 1:Nd, 'UniformOutput', false);
else,      Y = {@() Y0};  end

% Layout of the covariance bases
if ~ismatrix(Q),        layout = 'F';    % Full
elseif size(Q,2) > M,   layout = 'S';    % Compact symmetric
else,                   layout = 'D';    % Compact diagonal
end

% If full, make symmetric (simplifies later steps)
Q0 = Q; QS = Q; QF = Q;
if layout == 'F'
    QS     = spmb_full2sym(Q,2);
    layout = 'S';
elseif layout == 'S'
    QF     = spmb_sym2full(Q,2);
end

% -------------------------------------------------------------------------
% Initial fit + homoscedastic variance
B = cell(Nz,1);
h = 0;
N = 0;
for z=1:Nz
    Yz   = Y{z}();
    Yz   = Yz(all(isfinite(Yz) & Yz ~= 0,2),:);
    N    = N + size(Yz,1);
    B{z} = gllm_fit(Yz,X,1,opt.fit);
    R    = exp(B{z}*X') - Yz;
    h    = h + dot(R(:),R(:));
end
h = h / (M*N);

% Initialize hyperparameter (minimize KL between C and s^2 * I)
h = init_cov(QS,h,layout);

% Adapt mode for weighted gllm_fit
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
    R = 0;
    if ~USE_SYM && layout ~= 'D', A = spmb_sym2full(A); end
    for z=1:Nz
        Yz           = Y{z}();
        Yz           = Yz(all(isfinite(Yz) & Yz ~= 0,2),:);
        opt.fit.init = B{z};
        B{z}         = gllm_fit(Yz,X,spm_unsqueeze(A,1),opt.fit);
        R            = R + residual_matrix(Yz,X,B{z},A);
    end
    if USE_SYM && layout ~= 'D', A = spmb_sym2full(A); end
    R = R / N;

   
    % ---------------------------------------------------------------------
    % Utility matrices
    if layout == 'D',   P = diag(A) - (A .* R) .* A';
                        L = 2 * (A .* R) - eye(M);
    else,               P = A - (A * R) * A;
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

% Return model parameters
if nargout > 1
    B0 = B{1};
    if Nd == 1
        Yz           = Y{z}();
        B            = zeros([size(Yz,1) K], class(B0));
        msk          = all(isfinite(Yz) & Yz ~= 0,2);
        Yz           = Yz(msk,:);
        opt.fit.init = B0;
        B(msk,:)     = gllm_fit(Yz,X,spm_unsqueeze(A,1),opt.fit);
        B            = reshape(B, [Yshape(1:Nd) K]);
    else
        B  = zeros([Yshape(1:Nd) K], class(B{1}));
        for z=1:Nz
            Yz           = Y{z}();
            Bz           = zeros([size(Yz,1) K], class(B0{z}));
            msk          = all(isfinite(Yz) & Yz ~= 0,2);
            Yz           = Yz(msk,:);
            opt.fit.init = B0{z};
            B0{z}         = [];
            Bz(msk,:)    = gllm_fit(Yz,X,spm_unsqueeze(A,1),opt.fit);
            Bz           = reshape(Bz, [Yshape(1:Nd-1) 1 K]);
            B            = spm_subsasgn(B,Bz,z,Nd);
        end
    end
end

% Assign posterior precision to var P
P = H;
end

% =========================================================================
function R = residual_matrix_sym(Y,X,B,A)

M  = size(Y,2);
K  = size(X,2);

% Fitted signal
% -------------
Z  = exp(B*X');

% 2nd moment in Z           (N x M*(M+1)/2)
% ---------------
ZZ = spmb_sym_outer(Z,'dim',2);

% Precision about B         (N x K*(K+1)/2)
% -----------------
S  = ZZ;
A  = spm_unsqueeze(A,1);
if size(A,2) == M, S(:,1:M) = S(:,1:M) .* A;
else,              S = S .* A;  end
S  = spmb_sym_outer(spm_unsqueeze(X',1), S, 'dim', 2);

% Uncertainty about B       (N x K*(K+1)/2)
% -------------------
S(:,1:K) = S(:,1:K) + max(abs(S(:,1:K)),[],2) * 1e-8;
S  = spmb_sym_inv(S, 'dim', 2);

% Uncertainty about BX'     (N x M*(M+1)/2)
% ---------------------
S  = spmb_sym_outer(spm_unsqueeze(X,1), S, 'dim', 2);

% Expected moments          (M x M)
% ----------------
S  = exp(S);
ZZ = spm_squeeze(dot(ZZ, S, 1), 1);                % E[Z'*Z]
ZZ = spmb_sym2full(ZZ);
YZ = Z .* sqrt(S(:,1:M));                          % E[Z]
YZ = Y'*YZ;                                        % E[Z'*Y]

% Build residual matrix     (M x M)
% ---------------------
% R = E[(Z-Y)' * (Z-Y)]
YZ = YZ + YZ';
YY = Y' * Y;
R  = ZZ + YY - YZ;
end

% =========================================================================
function R = residual_matrix_full(Y,X,B,A)

M  = size(Y,2);
K  = size(X,2);
dM = spm_diagind(M);
A  = spm_unsqueeze(A,1);

% Fitted signal
% -------------
Z  = exp(B*X');

% 2nd moment in Z           (N x M x M)
% ---------------
ZZ = spm_unsqueeze(Z,2) .* spm_unsqueeze(Z,3);

% Precision about B         (N x K x K)
% -----------------
S = ZZ;
if size(A,2) == M, S(:,dM) = S(:,dM) .* A;
else,              S       = S       .* A;  end
S = outer(S,X');

% Uncertainty about B       (N x K x K)
% -------------------
S        = spmb_full2sym(S,2);
S(:,1:K) = S(:,1:K) + max(abs(S(:,1:K)),[],2) * 1e-8;
S        = spmb_sym_inv(S,2);
S        = spmb_sym2full(S,2);

% Uncertainty about BX'     (N x M x M)
% ---------------------
S = outer(S,X);

% Expected moments          (M x M)
% ----------------
S  = exp(S);
ZZ = spm_squeeze(dot(ZZ, S, 1), 1);                % E[Z'*Z]
YZ = Z .* sqrt(S(:,spm_diagind(M)));               % E[Z]
YZ = Y'*YZ;                                        % E[Z'*Y]

% Build residual matrix     (M x M)
% ---------------------
% R = E[(Z-Y)' * (Z-Y)]
YZ = YZ + YZ';
YY = Y' * Y;
R  = ZZ + YY - YZ;
end

function XAX = outer(A,X)
Xt  = spm_unsqueeze(X',1);
X   = spm_unsqueeze(X,1);
XA  = spmb_matmul(X,A,2);
XAX = spmb_matmul(XA,Xt,2);
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