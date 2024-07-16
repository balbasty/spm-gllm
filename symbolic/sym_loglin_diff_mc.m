clear
% Derivatives of multi-compartment loglinear model, assuming a different
% parameter set per component (B -> K x N x C)
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
B   = sym('B', [K N C], 'real');  % Model parameters
Y   = sym('Y', [M N 1], 'real');  % Observations 
A   = sym('A', [M M 1], 'real');  % Inverse covariance matrix
A   = A + A';                     % (symmetric)
% -------------------------------------------------------------------------
% Forward model
% -------------
Z   = sym([]);
for c=1:C
    Z(:,:,c) = exp(X(:,:,c) * B(:,:,c)); 
end
F   = sum(Z,3);                 % Model fit
R   = F - Y;                    % Residuals
NLL = 0.5 * trace(R*R'*A);      % Negative log-likelihood (terms in B)

% -------------------------------------------------------------------------
% Compute gradient
% ----------------
G = sym([]);
for c=1:C
for k=1:K
for n=1:N
    G(k,n,c) = diff(NLL, B(k,n,c));
end
end
end

% Analytical gradient
% -------------------
GG = sym([]);
for c=1:C
    GG(:,:,c) = X(:,:,c)' * (Z(:,:,c) .* (A * R));
end

checkG = simplify(G-GG, 100)
assert(~any(checkG(:)));

% -------------------------------------------------------------------------
% Compute Hessian
% ---------------
H = sym([]);
for c=1:C
for d=1:C
for k=1:K
for l=1:K
    H(k,l,c,d) = diff(G(k,1,c), B(l,1,d));
end
end
end
end
z  = reshape(Z(:,1,:), [M C]);
f  = F(:,1);
r  = R(:,1);

% Analytical Hessian
% ------------------
HH = sym([]);
for c=1:C
for d=c+1:C
    HH(:,:,c,d) = X(:,:,c)' * ((z(:,c)*z(:,d)') .* A) * X(:,:,d);
    HH(:,:,d,c) = HH(:,:,c,d)';
end
    HH(:,:,c,c) = X(:,:,c)' * ((z(:,c)*z(:,c)') .* A) * X(:,:,c);
    HH(:,:,c,c) = HH(:,:,c,c) + X(:,:,c)' *  diag(z(:,c) .* (A * r)) * X(:,:,c);
end

checkH = simplify(H-HH, 1000)
assert(~any(checkH(:)));

