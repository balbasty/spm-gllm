function R = sym_chol(A)
% Cholesky decomposition of a symmetric stored in compact form
%
% FORMAT R = sym_chol(A)
%
% A - (N*(N+1)/2 x 1) Compact positive-definite matrix
% R - (N*(N+1)/2 x 1) Compact triangular matrix
%__________________________________________________________________________
%
% A symmetric matrix stored in flattened form contains the diagonal first,
% followed by each column of the lower triangle
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ d b f ]
%                                       [ e f c ]
%
% Similarly, an upper or lower triangular matrix can be stored in flattened
% form:
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ 0 b f ]
%                                       [ 0 0 c ]
%__________________________________________________________________________

R  = A;
N2 = size(A,1);
N = findK(N2);
idx = mapidx(N);

sm0 = 1e-7 * (sum(A(1:N)) + 1e-40);
sm0 = sm0 * sm0;


for i=1:N

    sm = A(i);
    for k=i-1:-1:1
        sm = sm - R(idx(i,k)).^2;
    end
    if(sm <= sm0), sm = sm0; end
    R(i) = sqrt(sm);

    for j=i:N
        sm = A(idx(i,j));
        for k=i-1:-1:1
            sm = sm - R(idx(i,k))*R(idx(j,k));
        end
        R(idx(j,i)) = sm / R(i);
    end

    R(i) = A(i) / R(i);
end
end

% =========================================================================
function K = findK(K2)
K = (sqrt(1 + 8*K2) - 1)/2;
end

% =========================================================================
function idx = mapidx(K)
idx = zeros(K, 'uint64');
k = K+1;
for i=1:K
    idx(i,i) = i;
    for j=i+1:K 
        idx(i,j) = k;
        idx(j,i) = k;
        k = k + 1;
    end
end
end