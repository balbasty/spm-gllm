% -------------------------------------------------------------------------
% Domain size
% -----------
M = 2;
J = 2;
% -------------------------------------------------------------------------
% Input matrices
% --------------
h = sym('J', [J 1],   'real');
Q = sym('C', [J M M], 'real');
Q = Q + permute(Q, [1 3 2]);
% -------------------------------------------------------------------------
% Forward model
% -------------
C = squeeze(sum(repmat(h, [1 M M]) .* Q, 1));
A = inv(C);
I = eye(M);
% -------------------------------------------------------------------------
% Kullback-Leibler divergence
% ---------------------------
KL = trace(C) - log(det(C)) - M;
% -------------------------------------------------------------------------
% Compute gradient
% ----------------
fprintf('Compute gradient\n');
G  = sym([]);
GG = sym([]);
for j=1:J
    G(j)  = diff(KL, h(j));
    GG(j) = trace((I - A) * squeeze(Q(j,:,:)));
end
simplify(G-GG, 100)
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
        A * squeeze(Q(i,:,:)) * A * squeeze(Q(j,:,:)) ...
    );
end
end
fprintf('simplify...\n');
simplify(H-HH, 100)