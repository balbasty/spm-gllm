clear
% Derivatives of multi-compartment loglinear model, assuming a shared
% parameter set across components (B -> K x N x 1) and a diagonal
% residual covariance (A -> M x 1).
% -------------------------------------------------------------------------
% Domain size
% -----------
N   = 1;                        % Number of voxels
M   = 3;                        % Number of model variables
K   = 2;                        % Number of model parameters
C   = 2;                        % Number of components
% -------------------------------------------------------------------------
% Input matrices
% --------------
X   = sym('X', [M K C], 'real');  % Design matrix
B   = sym('B', [K N 1], 'real');  % Model parameters
Y   = sym('Y', [M N 1], 'real');  % Observations 
A   = sym('A', [M 1 1], 'real');  % Inverse covariance matrix
% -------------------------------------------------------------------------
% Forward model
% -------------
Z   = sym([]);
for c=1:C
    Z(:,:,c) = exp(X(:,:,c) * B); 
end
F   = sum(Z,3);                  % Model fit
R   = F - Y;                     % Residuals
NLL = 0.5 * trace(R*R'*diag(A)); % Negative log-likelihood (terms in B)

% -------------------------------------------------------------------------
% Compute gradient
% ----------------
G = sym([]);
for k=1:K
for n=1:N
    G(k,n) = diff(NLL, B(k,n));
end
end

% Analytical gradient
% -------------------
GG = sym([]);
for c=1:C
    GG(:,:,c) = X(:,:,c)' * (Z(:,:,c) .* (A .* R));
end
GG = sum(GG,3);

checkG = simplify(G-GG, 100)
assert(~any(checkG(:)));

% -------------------------------------------------------------------------
% Compute Hessian
% ---------------
H = sym([]);
for k=1:K
for l=1:K
    H(k,l) = diff(G(k,1), B(l,1));
end
end
z  = reshape(Z(:,1,:), [M C]);
f  = F(:,1);
r  = R(:,1);

% Analytical Hessian
% ------------------
HH = sym(zeros([K K]));
for c=1:C
    HH = HH + X(:,:,c)' * diag(A .* z(:,c).*z(:,c)) * X(:,:,c);
    % HH = HH + X(:,:,c)' *  diag(z(:,c) .* (A .* r)) * X(:,:,c);
for d=c+1:C
    H1 = X(:,:,c)' * diag(A .* z(:,c).*z(:,d)) * X(:,:,d);
    HH = HH + H1 + H1';
end
end

checkH = simplify(H-HH, 1000)
assert(~any(checkH(:)));

