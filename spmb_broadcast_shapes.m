function [shape] = spmb_broadcast_shapes(varargin)
% Broadcast left and right batch shapes
% (Utility for batched spmb_* functions)
%
% FORMAT shape = spmb_broadcast_shapes(shapes...,side)
% 
% shapes - (vector)           Shapes to broadcast
% side   - ('left'|'right')   Left or right implicit singleton padding
% shape  - (vector)           Broadcasted shape
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
    shapes = cellfun(@(x) [ones(1,max(0,n-length(x))) x], shapes);
else
    shapes = cellfun(@(x) [x ones(1,max(0,n-length(x)))], shapes);
end

if ~isempty(shapes)
    shape  = shapes{1};
    shapes = shapes(2:end);
    while ~isempty(shapes)
        shape  = max(shape, shapes{1});
        shapes = shapes(2:end);
    end
else
    shape = [];
end

end
