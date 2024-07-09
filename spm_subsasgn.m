function A = spm_subsasgn(A,V,i,d)
% Index-assign an array along a specified dimension (or dimensions)
%   A(i) = V
%
% FORMAT S = spm_slice_array(A,V,i,d)
%
% A - (array)            Multidimensional array (output)
% V - (array)            Multidimensional array (input value)
% i - (cell of vectors)  Indices
% d - (scalar or vector) Dimensions
%__________________________________________________________________________

% Yael Balbastre

s = repmat({':'}, 1, max(ndims(A),max(d)));
d = [d ones(1,max(0,numel(i)-length(d)))];

for n=1:length(i)
    s{d(n)} = i{n};
end

A(s{:}) = V;