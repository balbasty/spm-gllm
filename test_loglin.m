PLOT = false;
B0   = 1000;

% -------------------------------------------------------------------------
% model
T = 0.1:0.1:1;              % Sampled time points
O = ones(1, length(T));     
Z = zeros(1, length(T));
X = [O' Z' Z' -T'           % Design matrix
     Z' O' Z' -T'
     Z' Z' O' -T'];

BB = [2 2.2 2.4 1];         % True parameters (3x intercept + decay rate)
SD = 0.2;                   % Noise standard deviation
Y0 = exp(BB*X');            % Noise-free signal

% -------------------------------------------------------------------------
% noisy data
Y  = Y0 + SD * randn(size(Y0));

if PLOT
    figure
    hold on
    for b=B(1:end-1)
        TT = linspace(0, 1.5, 256);
        plot(TT, exp(b - TT))
    end
    scatter([T T T],Y','k')
    hold off
end

% -------------------------------------------------------------------------
% logfit OLS
B1 = 0;
B2 = 0;
for b=1:B0
    Y  = Y0 + SD * randn(size(Y0));
    B = gllm_logfit(Y,X,1,struct('iter',0));
    B1 = B1 + B;
    B2 = B2 + B'*B;
end

out.logfit.BIAS = B1/B0 - BB;
out.logfit.MSE = (B2 + B0*(BB'*BB) - BB'*B1 - B1'*BB)/B0;

out.logfit.B1 = B1 / B0;
out.logfit.B2 = B2 / B0;
out.logfit.B2 = B2 - B1'*B1;


% -------------------------------------------------------------------------
% logfit IRLS
B1 = 0;
B2 = 0;
for b=1:B0
    Y  = Y0 + SD * randn(size(Y0));
    B = gllm_logfit(Y,X,1,struct('iter',3));
    B1 = B1 + B;
    B2 = B2 + B'*B;
end

out.logfit3.BIAS = B1/B0 - BB;
out.logfit3.MSE = (B2 + B0*(BB'*BB) - BB'*B1 - B1'*BB)/B0;

out.logfit3.BB1 = B1 / B0;
out.logfit3.BB2 = B2 / B0;
out.logfit3.BB2 = B2 - B1'*B1;


% -------------------------------------------------------------------------
% nlfit
B1 = 0;
B2 = 0;
for b=1:B0
    Y  = Y0 + SD * randn(size(Y0));
    B = gllm_fit(Y,X);
    B1 = B1 + B;
    B2 = B2 + B'*B;
end

out.nlfit.BIAS = B1/B0 - BB;
out.nlfit.MSE = (B2 + B0*(BB'*BB) - BB'*B1 - B1'*BB)/B0;

out.nlfit.B1 = B1 / B0;
out.nlfit.B2 = B2 / B0;
out.nlfit.B2 = B2 - B1'*B1;

% -------------------------------------------------------------------------
fprintf('logfit  - bias = ['); fprintf('%12.6g ', 1e3*out.logfit.BIAS);       fprintf('] * 1e-3\n');
fprintf('logfit  - mse  = ['); fprintf('%12.6g ', 1e3*diag(out.logfit.MSE));  fprintf('] * 1e-3\n');
fprintf('logfit3 - bias = ['); fprintf('%12.6g ', 1e3*out.logfit3.BIAS);      fprintf('] * 1e-3\n');
fprintf('logfit3 - mse  = ['); fprintf('%12.6g ', 1e3*diag(out.logfit3.MSE)); fprintf('] * 1e-3\n');
fprintf('nlfit   - bias = ['); fprintf('%12.6g ', 1e3*out.nlfit.BIAS);        fprintf('] * 1e-3\n');
fprintf('nlfit   - mse  = ['); fprintf('%12.6g ', 1e3*diag(out.nlfit.MSE));   fprintf('] * 1e-3\n');