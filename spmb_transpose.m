function varargout = spmb_transpose(varargin)
% Transpose a batch of matrices
%
% FORMAT At = spmb_transpose(A)
% FORMAT At = spmb_transpose(A,i,j)
%
% A  - (M x N)  Input batch of matrices
% i  - (int)    First dimension to transpose [DIM > 0 ? DIM   : DIM-1]
% j  - (int)    Other dimension to transpose [DIM > 0 ? DIM+1 : DIM]
% At - (N x M)  Output batch of matrices
%__________________________________________________________________________
%
% FORMAT spmb_transpose(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (M x N x ...)
%           At  will have shape (N x M x ...)
%
%      * If `-1`:
%           A should have shape (... x M x N)
%           At  will have shape (... x N x M)
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
[varargout{1:nargout}] = do_transpose(dim,args{:});

end

function At = do_transpose(d,A,i,j)

if nargin < 3
    if d > 0
        i = d;
        j = d+1;
    else
        i = ndims(A)+d;
        j = ndims(A)+d+1;
    end
elseif nargin == 3
    error("spmb_transpose should take either 1 or 3 arguments")
end

% Accept negative "python-like" indices
i(i < 0) = ndims(A) + 1 + i(i < 0);
j(j < 0) = ndims(A) + 1 + j(j < 0);

dims = 1:ndims(A);
dims([i j]) = dims([j i]);
At = permute(A, dims);
end

