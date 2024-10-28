load('toydata.mat')
B0 = B;
S0 = S;

NONSPHERICITY = 1;  % (0=none,1=TE)

% ----------------------------------------
% Simulate observations (no nonsphericity)
X = [ones(numel(TE),1) -TE'];
Y = exp(B * X');

switch NONSPHERICITY
    case 0, Y = Y + randn(size(Y)) .* sqrt(S);
    case 1, Y = Y + randn(size(Y)) .* sqrt(S) .* (1 + 0.01 * TE);
end

% -------------------------------------------------------------------------
% Build design matrix and covariance basis
ORDER = 6;
X = (TE') .^ (0:ORDER);
Q = (TE') .^ (0:1);

N = size(B,1);
M = size(X,1);
K = size(X,2);

% Make sparse basis for spm_reml
Q = {spdiags(Q(:,1),0,M,M) spdiags(Q(:,2),0,M,M)};

% ------------------------------------
% Prepare observed covariance for ReML
B  = Y / X';
R  = B*X' - Y;
S  = dot(R,R,2) / (M-K);
RR = (R' * (R ./ S)) / N;
YY = Y ./ sqrt(S);
YY = (YY' * YY) / N;
U  = X*inv(X'*X)*X';

% ---------------------------
% Look at variance components

f = figure;
f.Position = [100 100 1024 256];
t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
nexttile
plot(diag(RR))
title('Residuals')
nexttile
plot(diag(U))
title('Uncertainty')
nexttile
plot(diag(RR+U))
title('Expected residuals')
ylim([0.85 1.15])

% -------------------------------------------------------------------------
% Estimate covariance
[C,h] = spm_reml(YY,X,Q);

% -------------------------------------------------------------------------
% Look at fits
figure
idx = 1;
hold off
scatter(TE, Y(idx,:));
hold on
T = linspace(0,18,128); 
plot(T, B(idx,:)*(T' .^ (0:ORDER))'); 
hold off;
