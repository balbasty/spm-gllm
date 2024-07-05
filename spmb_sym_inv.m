function A = spmb_sym_inv(A,varargin)
% Invert a batch of (compact) positive-definite linear systems
%
% FORMAT iA = spmb_sym_inv(A)
% 
% A  - (N*(N+1)/2 x 1) Compact positive-definite matrix
% iA - (N*(N+1)/2 x 1) Inverse compact matrix
%__________________________________________________________________________
%
% FORMAT spmb_sym_solve(A,DIM)
% FORMAT spmb_sym_solve(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape   (N2 x ...)
%           iA  will have shape   (N2 x ...)
%
%      * If `-1`:
%           A should have shape (... x N2)
%           iA  will have shape (... x N2)
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
A = spmb_sym_cholinv(R,varargin{:});