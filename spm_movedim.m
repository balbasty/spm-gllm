function A = spm_movedim(A,i,j)
% Move an array dimension
%
% FORMAT At = spm_movedim(A)
% FORMAT At = spm_movedim(A,src,dst)
%
% A   - (array)  Input batch of matrices
% src - (int)    Source position of the dimension to move
% dst - (int)    Target position of the dimension
% At  - (array)  Output batch of matrices
%__________________________________________________________________________

% Yael Balbastre

% Accept negative "python-like" indices
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

dims = 1:ndims(A);
dims(i) = [];
dims = [dims(1:j-1) i dims(1:j)];
A = permute(A, dims);
end

