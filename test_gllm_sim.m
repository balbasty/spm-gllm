load('toydata.mat')
B0 = B;
S0 = S;

NONSPHERICITY = 1;  % (0=none,1=TE)

% -------------------------------------------------------------------------
% Build design matrix and covariance basis
X = [ones(numel(TE),1) -TE'];
Q = [ones(numel(TE),1) TE']';

N = size(B,1);
M = size(X,1);
K = size(X,2);

% ----------------------------------------
% Simulate observations (no nonsphericity)
Y = exp(B * X');

switch NONSPHERICITY
    case 0, Y = Y + randn(size(Y)) .* sqrt(S);
    case 1, Y = Y + randn(size(Y)) .* sqrt(S) .* (1 + 0.01 * TE);
end

% ---------------------------------------------------------------------
% Estimate covariance
[C,~,h] = gllm_reml(Y,X,Q,struct('verb',2));

if false
ER = R;
Z  = exp(B*X');
RR = (Z-Y)' * (Z-Y);
RR = RR / N;

f = figure;
f.Position = [100 100 1024 256];
t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
nexttile
plot(diag(RR))
title('Residuals')
nexttile
plot(diag(ER-RR))
title('Uncertainty')
nexttile
plot(diag(ER))
ylim([0.9, 1.1])
title('Expected residuals')
end

% -------------------------------------------------------------------------
% Look at fits
Y0 = Y;

N  = size(Y0,1);
B  = gllm_fit(Y0,X,1./C);
R  = Y0 - exp(B*X');
S  = dot(R,R,2) / (M-K);
RR = (R' * (R ./ S)) / N;

figure
idx = 1;
hold off
scatter(TE, Y0(idx,:));
hold on
T = linspace(0,18,128); 
plot(T, exp(B(idx,:)*((-T)' .^ (0:1))')); 
hold off;

