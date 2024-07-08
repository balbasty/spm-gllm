function varargout = spmb_trace(varargin)
% Efficient batched trace
%
% FORMAT t = spmb_trace(A)
% A - (N x N)  Input batch of square matrices
% t - (1 x 1)  Trace of A
%
% FORMAT t = spmb_trace(A,B)
% A - (N x M)  Input batch of square matrices
% B - (M x N)  Input batch of square matrices
% t - (1 x 1)  Trace of A*B
%__________________________________________________________________________
%
% FORMAT spmb_trace(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (N x M x ...)
%           B should have shape (M x N x ...)
%           t   will have shape (1 x 1 x ...)
%
%      * If `-1`:
%           A should have shape (... x N x M)
%           B should have shape (... x M x N)
%           t   will have shape (... x 1 x 1)
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if length(args) == 1
    [varargout{1:nargout}] = trace_A(dim,args{:});
else
    [varargout{1:nargout}] = trace_AB(dim,args{:});
end

end

function t = trace_A(d,A)
% TODO: implement out own loop rather than using batch_diag to be faster
D = spmb_diag(A,'dim',d);
if d < 0
    d = d + ndims(A);
end
t = spm_unsqueeze(sum(D,d),d);
end

function t = trace_AB(d,A,B)
% TODO: can we do better than using batch_transpose?
AB = spmb_transpose(A,'dim',d) .* B;
if d < 0
    d = d + ndims(A);
end
t = sum(sum(AB,d),d+1);
end