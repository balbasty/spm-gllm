function x = spmb_sym_solve(A,y,varargin)
% Solve a batch of (compact) positive-definite linear systems
%
% FORMAT x = spmb_sym_solve(A,y)
% 
% A - (N*(N+1)/2 x 1) Compact positive-definite matrix
% y - (N x 1)         Input vector
% x - (N x 1)         Output vector
%__________________________________________________________________________
%
% FORMAT spmb_sym_solve(A,y,DIM)
% FORMAT spmb_sym_solve(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape   (N2 x ...)
%           x should have shape    (N x ...)
%           y   will have shape    (N x ...)
%
%      * If `-1`:
%           A should have shape (... x N2)
%           x should have shape (... x N)
%           y   will have shape (... x N)
%__________________________________________________________________________
%
% A symmetric matrix stored in flattened form contains the diagonal first,
% followed by each column of the lower triangle
%
%                                       [ a d e ]
%       [ a b c d e f ]       =>        [ d b f ]
%                                       [ e f c ]
%__________________________________________________________________________

% Yael Balbastre

R = spmb_sym_chol(A,varargin{:});
x = spmb_sym_cholls(R,y,varargin{:});