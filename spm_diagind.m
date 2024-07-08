function I = spm_diagind(shape,k)
% Return the linear indices of the diagonal elements of a matrix
%
% FORMAT I = spm_diagind(N)
% FORMAT I = spm_diagind([N M])
% FORMAT I = spm_diagind(...,k)
%
% N - (int)   Matrix size (first dimension)
% M - (int)   Matrix size (second dimention)                      [N]
% k - (int)   Diagonal to index (0: main, >0: upper, <0: lower)   [0]
% I - (K x 1) Linear indices of the diagonal
%__________________________________________________________________________

% Yael Balbastre

if nargin < 2, k = 0; end

if isempty(shape) || length(shape) > 2
    error("Shape must have one or two elements");
elseif length(shape) == 1
    N = shape(1);
    M = N;
else
    N = shape(1);
    M = shape(2);
end

if k < 0, L = M + k;
else,     L = N - k; end
L = min(L,min(M,N));

I = max(1,1-k);
I = (I:I+L-1)';
I = I + (I+k-1) * N;
