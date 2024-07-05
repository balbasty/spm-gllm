function [dim,args] = spmb_parse_dim(varargin)
% Utility function for parsing the index of the nonbatch dimension and
% dispatching it to a lower level implementation.
%
% This parses arguments of the form `func(..., 'dim', DIM)`.
%
% * If `DIM` is a positive integer, this assumes that the input has:
%   - exactly `DIM-1` batch dimensions on the left;
%   - followed by one or more dimensions of interest (their number depends 
%     on the underlying function; could be one for a batch of vectors, 
%     two for a batch of matrices, etc);
%   - followed by any number of additional batch dimensions on the right.
%
% * If `DIM` is a negative integer, this assumes that the input has:
%   - exactly `-DIM-1` batch dimensions on the right;
%   - preceded by one or more dimensions of interest;
%   - preceded by any number of additional batch dimensions on the left.
%
% The number of dimensions of interest depends on the underlying function; 
% could be one for a batch of vectors, two for a batch of matrices, etc.
%
% Importantly, batch dimensions (left or right) of multiple inputs get 
% automatically broadcasted according to matlab broadcasting rules.
%
% If `DIM` is negative, the leading (left) batch dimensions get left-padded
% with singleton (as in Python's conventions), whereas if `DIM` is 
% positive, the trailing (right) batch dimensions get right-padded with 
% singleton (as is always the case in Matlab).
%
% ---
%
% For example a matrix and vector with `DIM=3` could have shapes:
%
%   matrix: [ A  1  M  N  C  D ]     vector: [ 1  B  N  C ]
%             ----  ----  ----                 ---- --- -
%             lead   mat  trail                lead vec trail
%
% And the resulting (e.g.) matrix-vector product would have shape:
%
%  matrix * vector: [ A  B  M  C  D ]
%                     ---- --- ----
%                     lead vec trail
% ---
%
% If we have `DIM=-3` instead:
%
%   matrix: [ D  C  M  N  1  A ]     vector: [ C  N  B  1 ]
%             ----  ----  ----                 - --- ----
%             lead   mat  trail             lead vec trail
%
% And the resulting (e.g.) matrix-vector product would have shape:
%
%  matrix * vector: [ D  C  M  B  A ]
%                     ---- --- ----
%                     lead vec trail
%
% ---
%
% In general, `DIM=1` follows Matlab conventions whereas `DIM=-1` follows 
% Python's convention.
%__________________________________________________________________________

% Yael Balbastre

dim = 1;
for i=nargin-1:-1:1
    key = varargin{i};
    if ischar(key) && strcmpi(key, 'dim')
        dim = varargin{i+1};
        varargin(i:i+1) = [];
        break
    end
end
args = varargin;

if dim == 0
    error('Index of first nonbatch dimension cannot be zero');
end

end