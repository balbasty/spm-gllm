function [left,item,right] = spmb_splitsize(X,d,n)
% Splits shape beween left-batch, item and right-batch components
% (Utility for batched spmb_* functions)
%
% FORMAT [left,item,right] = spmb_splitsize(X,d,n)
%
% X - (array) Batch of items
% d - (int)   Index of first (if > 0) or last (if < 0) "item" dimension
% n - (int)   Number of item dimensions
%
% left  - (vector) Left batch shape
% item  - (vector) Item shape
% right - (vector) Right batch shape
%__________________________________________________________________________

% Yael Balbastre

if d < 0, d = size(X)+2+d-n; end
shape = size(X);
left  = shape(1:d-1);
item  = shape(d:d+n-1);
right = shape(d+n:end);
end