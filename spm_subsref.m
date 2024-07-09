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

s = repmat({':'}, 1, max(ndims(A),max(d)));
d = [d ones(1,max(0,numel(i)-length(d)))];

for n=1:length(i)
    s{d(n)} = i{n};
end

V = A(s{:});