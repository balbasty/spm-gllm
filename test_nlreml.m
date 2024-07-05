PLOT = false;

% -------------------------------------------------------------------------
% model
T = 0.1:0.1:1;              % Sampled time points
O = ones(1, length(T));     
Z = zeros(1, length(T));
X = [O' Z' Z' -T'           % Design matrix
     Z' O' Z' -T'
     Z' Z' O' -T'];

N = 36864;
M = size(X,1);
K = size(X,2);

BB = [2 2.2 2.4 1];         % True parameters (3x intercept + decay rate)
Y0 = exp(BB*X');            % Noise-free signal
SD = 0.2 * [T T T];         % Noise standard deviation

% -------------------------------------------------------------------------
% noisy data
Y  = Y0 + SD .* randn([N M]);

if PLOT
    figure
    hold on
    for b=BB(1:end-1)
        TT = linspace(0, 1.5, 256);
        plot(TT, exp(b - TT))
    end
    scatter([T T T],Y','k')
    hold off
end

% -------------------------------------------------------------------------
% nlreml
Y     = Y0 + SD .* randn([N M]);
[C,B] = gllm_reml(Y,X,[],struct('verb',2,'fit',struct('verb',0)));