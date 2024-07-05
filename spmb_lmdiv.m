function varargout = spmb_lmdiv(varargin)
% Compute the left matrix division between batches of matrices
% 
% !!! very inefficient implementation
%
% FORMAT C = spmb_lmdiv(A, B)
%
% A - (M x N) Input batch of left matrices
% B - (M x K) Input batch of right matrices
% C - (N x K) Output batch of matrices
%__________________________________________________________________________
%
% FORMAT spmb_lmdiv(...,'dim',DIM)
%
% DIM - (int) Index of first (>0: left, <0: right) nonbatch dimensions [1]
%
%      * If `1`:
%           A should have shape (M x N x ...)
%           B should have shape (M x K x ...)
%           C   will have shape (N x K x ...)
%           Batch dimensions are implicitely padded on the right
%
%      * If `-1`:
%           A should have shape (... x M x N)
%           B should have shape (... x M x K)
%           C   will have shape (... x N x K)
%           Batch dimensions are implicitely padded on the left
%
%      * See `spmb_parse_dim` for more details.
%__________________________________________________________________________

% Yael Balbastre

[dim,args] = spmb_parse_dim(varargin{:});
if dim > 0
    [varargout{1:nargout}] = left_lmdiv(dim,args{:});
else
    [varargout{1:nargout}] = right_lmdiv(dim,args{:});
end

end

% =========================================================================
% Matlab-style: matrix on the left (B x M x N x ...)
function X = left_lmdiv(d,A,Y)

asrow = isrow(Y);
if asrow, Y = reshape(Y, size(Y,2), size(Y,1)); end

M = size(A,d);
N = size(A,d+1);
K = size(Y,d+1);

Ashape = size(A);
Albatch = Ashape(1:d-1);
Arbatch = Ashape(d+2:end);

Yshape = size(Y);
Ylbatch = Yshape(1:d-1);
Yrbatch = Yshape(d+2:end);

Arbatch = [Arbatch ones(1, max(0, length(Yrbatch) - length(Arbatch)))];
Yrbatch = [Yrbatch ones(1, max(0, length(Arbatch) - length(Yrbatch)))];
Xrbatch = max(Arbatch,Yrbatch);
Xlbatch = max(Albatch,Ylbatch);

X = zeros([Xlbatch N K Xrbatch], class(Y(1)/A(1)));
A = reshape(A, [prod(Albatch) M N prod(Arbatch)]);
L = length(Xlbatch);

for i=1:prod(Albatch)
for j=1:prod(Arbatch)
    if M == N
        iA = inv(spm_squeeze(A(i,:,:,j), [1 4]));
    else
        iA = pinv(spm_squeeze(A(i,:,:,j), [1 4]));
    end

    if ~isempty(Albatch)
        XI = num2cell(ind2sub(Albatch, i));
        for k=1:length(XI)
            if Albatch(k) == 1
                XI{k} = ':';
            end
        end
        YI = XI;
        for k=1:length(XI)
            if Ylbatch(k) == 1
                YI{k} = 1;
            end
        end
    else
        XI = {};
        YI = {};
    end
    if ~isempty(Arbatch)
        XJ = num2cell(ind2sub(Arbatch, j));
        for k=1:length(XJ)
            if Arbatch(k) == 1
                XJ{k} = ':';
            end
        end
        YJ = XJ;
        for k=1:length(XJ)
            if Yrbatch(k) == 1
                YJ{k} = 1;
            end
        end
    else
        XJ = {};
        YJ = {};
    end
       
    iA = reshape(iA, [ones(1,L) N M]);
    X(XI{:},:,:,XJ{:}) = spmb_matmul(iA, Y(YI{:},:,:,YJ{:}), 'dim', L+1);
end
end

if asrow, X = reshape(X, size(X,2), size(X,1)); end

end

% =========================================================================
% Python-style: matrix on the right (... x M x N x B)
function X = right_lmdiv(d,A,Y)

ascol = iscolumn(Y);
if ascol, Y = reshape(Y, size(Y,2), size(Y,1)); end

M = size(A,ndims(A)+d);
N = size(A,ndims(A)+d+1);
K = size(Y,ndims(Y)+d+1);

Ashape = size(A);
Albatch = Ashape(1:ndims(A)+d-1);
Arbatch = Ashape(ndims(A)+d+2:end);

Yshape = size(Y);
Ylbatch = Yshape(1:ndims(Y)+d-1);
Yrbatch = Yshape(ndims(Y)+d+2:end);

Albatch = [ones(1, max(0, length(Ylbatch) - length(Albatch))) Albatch];
Ylbatch = [ones(1, max(0, length(Albatch) - length(Ylbatch))) Ylbatch];
Xrbatch = max(Arbatch,Yrbatch);
Xlbatch = max(Albatch,Ylbatch);

X = zeros([Xlbatch N K Xrbatch], class(Y(1)/A(1)));
A = reshape(A, [prod(Albatch) M N prod(Arbatch)]);
L = length(Xlbatch);

for i=1:prod(Albatch)
for j=1:prod(Arbatch)
    if M == N
        iA = inv(spm_squeeze(A(i,:,:,j), [1 4]));
    else
        iA = pinv(spm_squeeze(A(i,:,:,j), [1 4]));
    end

    if ~isempty(Albatch)
        XI = num2cell(ind2sub(Albatch,i));
        for k=1:length(XI)
            if Albatch(k) == 1
                XI{k} = ':';
            end
        end
        YI = XI;
        for k=1:length(XI)
            if Ylbatch(k) == 1
                YI{k} = 1;
            end
        end
    else
        XI = {};
    end
    if ~isempty(Arbatch)
        XJ = num2cell(ind2sub(Arbatch,j));
        for k=1:length(XJ)
            if Arbatch(k) == 1
                XJ{k} = ':';
            end
        end
        YJ = XJ;
        for k=1:length(XJ)
            if Yrbatch(k) == 1
                YJ{k} = 1;
            end
        end
    else
        XJ = {};
    end
    
    iA = reshape(iA, [ones(1,L) N M]);
    X(XI{:},:,:,XJ{:}) = spmb_matmul(iA,Y(XI{:},:,:,XJ{:}),'dim',L+1);
end
end

if ascol, X = reshape(X, size(X,2), size(X,1)); end

end