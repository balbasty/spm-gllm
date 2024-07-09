function V = spm_subsref(A,i,d)
% Index an array along a specified dimension (or dimensions)
%   V = A(i)
%
% FORMAT S = spm_slice_array(A,i,d)
%
% A - (array)            Multidimensional array
% i - (cell of vectors)  Indices
% d - (scalar or vector) Dimensions
% V - (array)            Output subarray
%__________________________________________________________________________

% Yael Balbastre

s = repmat({':'}, max(ndims(A),max(d)));
d = [d ones(max(0,length(A)-length(d)))];

for n=1:length(i)
    s{d(n)} = i{d};
end

V = A(s{:});