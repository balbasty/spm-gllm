% -------------------------------------------------------------------------
% Domain size
% -----------
N   = 1;                        % Number of voxels
M   = 1;                        % Number of model variables
K   = 1;                        % Number of model parameters
J   = 2;                        % Number of covariance bases
% -------------------------------------------------------------------------
% Input matrices
% --------------
% X   = sym('X', [M K],   'real');   % Design matrix
% B   = sym('B', [K N],   'real');   % Model parameters
% Y   = sym('Y', [M N],   'real');   % Observations 
R   = sym('R', [M N],   'real');   % Residuals 
h   = sym('h', [J 1],   'real');   % Covariance parameters
Q   = sym('Q', [J M M], 'real');   % Covariance bases
Q   = Q + permute(Q, [1 3 2]);     % (symmetric)
% -------------------------------------------------------------------------
% Forward model
% -------------
C   = squeeze(sum(repmat(h, [1 M M]) .* Q, 1));  % Covariance matrix
A   = inv(C);                   % Precision matrix
RR  = (R*R')/N;
NLL = trace(RR*A);          % Negative log-likelihood (terms in A)
NLL = NLL - log(det(A));

P   = A - A*RR*A;

% -------------------------------------------------------------------------
% Compute gradient
% ----------------
fprintf('Compute gradient\n');
G  = sym([]);
GG = sym([]);
for j=1:J
    G(j)  = diff(NLL, h(j));
    GG(j) = trace(P * squeeze(Q(j,:,:)));
end
fprintf('simplify...\n');
simplify(G-GG, 100)
% assert(~any(any(simplify(G-GG, 100))));

% -------------------------------------------------------------------------
% Compute Hessian
% ---------------
fprintf('Compute Hessian\n');
H  = sym([]);
HH = sym([]);
for i=1:J
for j=1:J
    H(i,j)  = diff(G(i), h(j));
    HH(i,j) = trace( ...
        A * squeeze(Q(i,:,:)) * A * squeeze(Q(j,:,:)) * ...
        (2 * A * RR - eye(M)) ...
    );
end
end
fprintf('simplify...\n');
simplify(H-HH, 100)
% assert(~any(any(simplify(H-HH, 100))));

% Fisher's scoring
% ----------------
H0 = subs(simplify(H),RR,inv(A));
HH0 = sym([]);
for i=1:J
for j=1:J
HH0(i,j) = trace(A * squeeze(Q(i,:,:)) * A * squeeze(Q(j,:,:)));
end
end
simplify(H0-HH0, 100)


