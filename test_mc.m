PLOT = false;
B0   = 1;

% -------------------------------------------------------------------------
% model
T = 0.1:0.1:3;              % Sampled time points
O = ones(1, length(T));     
Z = zeros(1, length(T));
X = cat(3, ...              % Design matrix
    [O' Z' Z' Z' Z' Z' Z' Z' -T' Z'
     Z' Z' O' Z' Z' Z' Z' Z' -T' Z'
     Z' Z' Z' Z' O' Z' Z' Z' -T' Z'
     Z' Z' Z' Z' Z' Z' O' Z' -T' Z'], ...          
    [Z' O' Z' Z' Z' Z' Z' Z' Z' -T'
     Z' Z' Z' O' Z' Z' Z' Z' Z' -T'
     Z' Z' Z' Z' Z' O' Z' Z' Z' -T'
     Z' Z' Z' Z' Z' Z' Z' O' Z' -T']);

M = size(X,1);
K = size(X,2);

BB = [2*([0.3 0.7]) 2*([0.7 0.3]) 2*[0.6 0.4] 2*[0.5 0.5] [1 0.2]];
SD = 0.1;

BX = spm_unsqueeze(BB,2).*spm_unsqueeze(X,1);
Y0 = sum(exp(sum(BX,3)),4);
BB = BB(:)';

% -------------------------------------------------------------------------
% noisy data
Y  = Y0 + SD * randn(size(Y0));

if PLOT
    colors = hsv((numel(BB)-2)/2);
    figure
    hold on
    for i=1:(numel(BB)-2)/2
        TT = linspace(0, 3, 256);
        plot(TT, exp(BB(2*(i-1)+1) - BB(end-1) * TT), '--', 'Color', colors(i,:))
        plot(TT, exp(BB(2*(i-1)+2) - BB(end)   * TT), ':', 'Color', colors(i,:))
        plot(TT, exp(BB(2*(i-1)+1) - BB(end-1) * TT) + ...
                 exp(BB(2*(i-1)+2) - BB(end)   * TT), 'Color', colors(i,:))
        scatter(T, Y((i-1)*numel(T)+1:i*numel(T)), [], colors(i,:))
    end
    hold off
end

% -------------------------------------------------------------------------
% nlfit
BB = BB(:);
B1 = 0;
B2 = 0;
for b=1:B0
    Y  = Y0 + SD * randn(size(Y0));
    B = gllm_fit(Y,X,1,struct('verb',2));
    B  = B(:);
    B1 = B1 + B;
    B2 = B2 + B'*B;
end

out.nlfit.BIAS = B1/B0 - BB;
out.nlfit.MSE = (B2 + B0*(BB'*BB) - BB'*B1 - B1'*BB)/B0;

out.nlfit.B1 = B1 / B0;
out.nlfit.B2 = B2 / B0;
out.nlfit.B2 = B2 - B1'*B1;

% -------------------------------------------------------------------------
fprintf('nlfit   - bias = ['); fprintf('%12.6g ', 1e3*out.nlfit.BIAS);        fprintf('] * 1e-3\n');
fprintf('nlfit   - mse  = ['); fprintf('%12.6g ', 1e3*diag(out.nlfit.MSE));   fprintf('] * 1e-3\n');

if true
    colors = hsv((numel(BB)-2)/2);
    figure

    subplot(1,2,1)
    hold on
    for i=1:(numel(BB)-2)/2
        TT = linspace(0, 3, 256);
        plot(TT, exp(BB(2*(i-1)+1) - BB(end-1) * TT), '--', 'Color', colors(i,:))
        plot(TT, exp(BB(2*(i-1)+2) - BB(end)   * TT), ':', 'Color', colors(i,:))
        plot(TT, exp(BB(2*(i-1)+1) - BB(end-1) * TT) + ...
                 exp(BB(2*(i-1)+2) - BB(end)   * TT), 'Color', colors(i,:))
        scatter(T, Y((i-1)*numel(T)+1:i*numel(T)), [], colors(i,:))
    end
    hold off

    subplot(1,2,2)
    hold on
    for i=1:(numel(BB)-2)/2
        TT = linspace(0, 3, 256);
        plot(TT, exp(B(2*(i-1)+1) - B(end-1) * TT), '--', 'Color', colors(i,:))
        plot(TT, exp(B(2*(i-1)+2) - B(end)   * TT), ':', 'Color', colors(i,:))
        plot(TT, exp(B(2*(i-1)+1) - B(end-1) * TT) + ...
                 exp(B(2*(i-1)+2) - B(end)   * TT), 'Color', colors(i,:))
        scatter(T, Y((i-1)*numel(T)+1:i*numel(T)), [], colors(i,:))
    end
    hold off
end

function B = clip(B)
B(:,end-2:end) = max(0, B(:,end-2:end));
end