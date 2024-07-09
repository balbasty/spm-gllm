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

s = repmat({':'}, max(ndims(A),max(d)));
d = [d ones(max(0,length(A)-length(d)))];

for n=1:length(i)
    s{d(n)} = i{d};
end

A(s{:}) = V;