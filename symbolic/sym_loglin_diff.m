% -------------------------------------------------------------------------
% Domain size
% -----------
N   = 1;                        % Number of voxels
M   = 3;                        % Number of model variables
K   = 2;                        % Number of model parameters
% -------------------------------------------------------------------------
% Input matrices
% --------------
X   = sym('X', [M K], 'real');  % Design matrix
B   = sym('B', [K N], 'real');  % Model parameters
Y   = sym('Y', [M N], 'real');  % Observations 
A   = sym('A', [M M], 'real');  % Inverse covariance matrix
A   = A + A';                   % (symmetric)
% -------------------------------------------------------------------------
% Forward model
% -------------
Z   = exp(X * B);               % Model fit
R   = Z - Y;                    % Residuals
NLL = 0.5 * trace(R*R'*A);      % Negative log-likelihood (terms in B)

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
GG = X' * (Z .* (A * R));

assert(~any(any(simplify(G-GG, 100))));

% -------------------------------------------------------------------------
% Compute Hessian
% ---------------
H = sym([]);
for k=1:K
    for l=1:K
        H(k,l) = diff(G(k,1), B(l,1));
    end
end
z  = Z(:,1);
r  = R(:,1);

% Analytical Hessian
% ------------------
HH = X' * ((z*z') .* A  + diag(z .* (A * r))) * X;

assert(~any(any(simplify(H-HH, 100))));

