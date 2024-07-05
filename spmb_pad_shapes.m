function [varargout] = spmb_pad_shapes(varargin)
% Implicit left/right expansion of shapes with singleton dimensions.
% (Utility for batched spmb_* functions)
%
% FORMAT [shapes...] = spmb_broadcast_shapes(shapes...,side)
% 
% shapes - (vector)           Shapes to pad
% side   - ('left'|'right')   Left or right implicit singleton padding
%__________________________________________________________________________

% Yael Balbastre

side = 'R';
if ischar(varargin{end})
    side = varargin{end};
    varargin = varargin(1:end-1);
end
side = upper(side(1));

shapes = varargin;
n = max(cellfun(@length,shapes));

if side == 'L'
    shapes = cellfun(@(x) [ones(1,max(0,n-length(x))) x], shapes, 'UniformOutput', false);
else
    shapes = cellfun(@(x) [x ones(1,max(0,n-length(x)))], shapes, 'UniformOutput', false);
end

varargout = shapes(1:nargout);

end