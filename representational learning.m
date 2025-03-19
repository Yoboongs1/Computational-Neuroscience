clear all;
load('representational.mat')

%% Data Visualisations
plotIm(Y(1:100,:)');
% Plot generative weights trained
plotIm(W);

%% 1a. Marginal Distribution p(xk)
% Compute values of latent variable x from data y using feed-forward weights w
X = Y * R;

% Define the specific components to analyze
k_values = [50, 100];
numK = length(k_values);

% Create a figure for plotting
fig = figure;
fig.Position = [100, 0, 2000, 800]; % Adjust figure size for better row alignment

% Loop through the selected components
for i = 1:numK
    k = k_values(i); % Current component
    xk = X(:, k); % Extract latent variable values for component k
    
    % Compute kurtosis for xk
    kurtosis_xk = compute_kurtosis(xk); % Kurtosis of xk

    % Compute mean and standard deviation of xk
    mu = mean(xk);
    sigma = std(xk);

    % Generate Gaussian and Laplace distributions for comparison
    x = -10:0.01:10;
    yGauss = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((x - mu) / sigma).^2);
    b = sigma / sqrt(2); % Scale parameter for Laplace distribution
    yLaplace = (1 / (2 * b)) * exp(-abs(x - mu) / b);

    % Ensure subplot layout is 1 row per component with 3 columns
    row = i; 
    % (1) Marginal distribution on normal scale
    ax1 = subplot(numK, 3, (row - 1) * 3 + 1);
    histogram(xk, 'Normalization', 'pdf');
    hold on;
    plot(x, yGauss, 'LineWidth', 1, 'DisplayName', 'N(μ_{xk},σ_{xk}^2)');
    plot(x, yLaplace, 'LineWidth', 1, 'DisplayName', 'Laplace(μ_{xk},b)');
    hold off;
    xlim([-10, 10]);
    xlabel(sprintf("x_{%d}", k));
    ylabel('Probability Density');
    title(sprintf('p(c_{%d}) | Kurtosis = %.2f', k, kurtosis_xk));
    legend;

    % (2) Marginal distribution on log scale
    ax2 = subplot(numK, 3, (row - 1) * 3 + 2);
    histogram(xk, 'Normalization', 'pdf');
    hold on;
    plot(x, yGauss, 'LineWidth', 1.5);
    plot(x, yLaplace, 'LineWidth', 1.5);
    hold off;
    xlim([-10, 10]);
    ylim([1e-20, 1]);
    set(gca, 'YScale', 'log');
    xlabel(sprintf("x_{%d}", k));
    ylabel('Log Probability Density');
    title(sprintf('log(p(x_{%d}))', k));

    % (3) Display the generative weights for the current component
    ax3 = subplot(numK, 3, (row - 1) * 3 + 3);
    genWeights = W(:, k);
    imagesc(reshape(genWeights, [sqrt(1024), sqrt(1024)]));
    axis square;
    xlabel(sprintf('W_{%d}', k));
    set(gca, 'yticklabel', '', 'xticklabel', '');
    box on;
    colormap gray;
    title(sprintf('Generative Weight for k = %d', k));

end

%% 1b. Joint and conditional distribution p(xk1, xk2), p(xk2|xk1)
% Define pairs of components to analyse
pairs = [63, 93; 1, 256];

% Create a figure for plotting
fig = figure;
fig.Position = [0, 30, 2000, 900];

% Define two slice values for conditional distribution
xk1Slice1 = 1; % First slice value
xk1Slice2 = 2; % Second slice value

% Define histogram edges
edges = linspace(-20, 20, 201);

% Loop through each pair
for pair = 1:size(pairs, 1)
    % Extract latent variables for the current pair
    xk1 = X(:, pairs(pair, 1));
    xk2 = X(:, pairs(pair, 2));
    
    % Compute the 2D histogram (joint distribution)
    histVals = histcounts2(xk1, xk2, edges, edges, 'Normalization', 'probability');

    % Compute the marginal distribution p(xk1)
    xk1Hist = histcounts(xk1, edges, 'Normalization', 'probability');

    % Compute the conditional distribution p(xk2 | xk1)
    condProbs = histVals ./ xk1Hist'; % Normalize each row
    condProbs(isnan(condProbs)) = 0; % Handle NaN values

    % Find the indices of the slices closest to xk1Slice1 and xk1Slice2
    [~, sliceIdx1] = min(abs(edges - xk1Slice1));
    [~, sliceIdx2] = min(abs(edges - xk1Slice2));

    % Plot the conditional probability heatmap
    ax1 = subplot(size(pairs, 1), 4, 1 + (pair-1)*4);
    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0, 0.1]);
    colormap(ax1, 'parula');
    hold on;
    line([edges(1), edges(end)], [edges(sliceIdx1), edges(sliceIdx1)], 'Color', 'yellow', 'LineWidth', 1.5);
    line([edges(1), edges(end)], [edges(sliceIdx2), edges(sliceIdx2)], 'Color', 'red', 'LineWidth', 1.5);
    hold off;
    axis square;
    xlim([-10, 10]);
    ylim([-10, 10]);
    xlabel('x_{k2}');
    ylabel('x_{k1}');
    colorbar;
    title(sprintf("p(x_{k2}|x_{k1}) for k1 = %d , k2 = %d", pairs(pair, 1), pairs(pair, 2)));

    % Plot the slices of xk1
    ax2 = subplot(size(pairs, 1), 4, 2 + (pair-1)*4);
    xk2Hist = histcounts(xk2, edges, 'Normalization', 'probability');
    plot(edges(1:end-1), xk2Hist, 'LineWidth', 1, 'DisplayName', 'p(x_{k2})');
    hold on;
    plot(edges(1:end-1), condProbs(sliceIdx1, :), 'LineWidth', 1, 'DisplayName', sprintf('p(x_{k2} | x_{k1}=%d)', edges(sliceIdx1)));
    plot(edges(1:end-1), condProbs(sliceIdx2, :), 'LineWidth', 1, 'DisplayName', sprintf('p(x_{k2} | x_{k1}=%d)', edges(sliceIdx2)));
    hold off;
    legend('Fontsize',10);
    grid on;
    set(gca, 'GridAlpha', 0.1);
    axis square;
    xlim([-10, 10]);
    ylim([0, 0.165]);
    xlabel('x_{k2}');
    ylabel('Probability Density');
    title(sprintf("Marginal and Conditional at x_{k1} = 1 and x_{k2} = 2"));

    % Plot generative weights for k1 and k2
    ax3 = subplot(size(pairs, 1), 4, 3 + (pair-1)*4);
    wk1 = W(:, pairs(pair, 1));
    imagesc(reshape(wk1, [sqrt(1024), sqrt(1024)]), max(abs(wk1)) * [-1, 1] + [-1e-5, 1e-5]);
    axis square;
    xlabel(sprintf('W_{k1} = W_{%d}', pairs(pair, 1)));
    set(gca, 'yticklabel', '', 'xticklabel', '');
    colormap(ax3, 'gray');
    box on;
    title(sprintf('Generative Weight for k1 = %d', pairs(pair, 1)));

    ax4 = subplot(size(pairs, 1), 4, 4 + (pair-1)*4);
    wk2 = W(:, pairs(pair, 2));
    imagesc(reshape(wk2, [sqrt(1024), sqrt(1024)]), max(abs(wk2)) * [-1, 1] + [-1e-5, 1e-5]);
    axis square;
    xlabel(sprintf('W_{k2} = W_{%d}', pairs(pair, 2)));
    set(gca, 'yticklabel', '', 'xticklabel', '');
    colormap(ax4, 'gray');
    box on;
    title(sprintf('Generative Weight for k2 = %d', pairs(pair, 2)));
end
%% 2. Find optimal parameters that minimise loss function
X = Y(1:32000,:) * R;
[N, K] = size(X); % N = 32000, K = 256

% Initialize log-transformed parameters
param= zeros(K*(K+1),1);
[param, fX, iteration] = minimize(param, 'log_likelihood' , 1000, X);

% Verify gradient computation using checkgrad
% epsilon = 1e-5;
%diff = checkgrad('log_likelihood', [logA(:); logb], epsilon, X);
%fprintf('Gradient check difference: %e\n', diff);

fig = figure;
fig.Position = [0, 30, 600, 400];

% Start from 2nd iteration as first two values are too large 
plot(2:iteration, fX(2:iteration));
xlabel('Iteration');
ylabel('Negative Conditional Log Likelihood');
grid;

%% 3a. Marginal Distribution p(ck)
% Compute normalized variables ck
A = exp(reshape(param(1:K^2), K, K));
A(1:K+1:end) = 0; % Set diagonal to zero (no self-dependencies)
b = exp(param(K^2+1:end)); % Bias term
sigma2 = X.^2 * A + repmat(b', N, 1); % Variance
sigma = sqrt(sigma2);
C = X ./ sigma; % Matrix containing normalised variables ck

% Create a figure for plotting
fig = figure;
fig.Position = [100, 0, 1200, 800];

% Loop over each k value
for i = 1:numK
    k = k_values(i); % Current component
    xk = X(:, k); % Original variable xk
    ck = C(:, k); % Normalised variable ck

    % Compute kurtosis for ck
    kurtosis_ck = compute_kurtosis(ck);

    % Compute Gaussian for ck
    mu_ck = mean(ck);
    sigma_ck = std(ck);
    x_ck = linspace(min(ck), max(ck), 1000);
    gauss_ck = (1 / sqrt(2 * pi * sigma_ck^2)) * exp(-0.5 * ((x_ck - mu_ck) / sigma_ck).^2);

    % Plot 1: Original histogram for ck with Gaussian overlay
    ax1 = subplot(numK, 2, 1 + (i-1)*2);
    histogram(ck, Normalization='pdf', BinWidth=0.1);
    hold on;
    plot(x_ck, gauss_ck, 'r', LineWidth=1.5, DisplayName='N(μ,σ^2)');
    hold off;
    xlabel(sprintf('c_{%d}', k));
    ylabel('Probability Density');
    title(sprintf('p(c_{%d}) | Kurtosis = %.2f', k, kurtosis_ck));
    legend;

    % Plot 2: Log plot for ck with Gaussian overlay (y-axis logarithmic)
    ax2 = subplot(numK, 2, 2 + (i-1)*2);
    histogram(ck, Normalization='pdf', BinWidth=0.1);
    hold on;
    plot(x_ck, gauss_ck, 'r', LineWidth=1.5, DisplayName='N(μ,σ^2)');
    hold off;
    set(gca, 'YScale', 'log'); % Set y-axis to log scale
    xlabel(sprintf('c_{%d}', k));
    ylabel('Log Probability Density');
    title(sprintf('log(p(c_{%d}))', k));
end

%% 3b. Joint and Conditional distribution p(ck1, ck2), p(ck2|ck1)
% Create a figure for plotting
fig = figure;
fig.Position = [0, 30, 1200, 800];

% Define two slice values for conditional distribution
ck1Slice1 = 1; % First slice value
ck1Slice2 = 2; % Second slice value

% Define histogram edges
edges = linspace(-20, 20, 201);

for pair = 1:size(pairs, 1)
    % Extract latent variables for the current pair
    ck1 = C(:, pairs(pair, 1));
    ck2 = C(:, pairs(pair, 2));
    
    % Compute the 2D histogram (joint distribution)
    histVals = histcounts2(ck1, ck2, edges, edges, 'Normalization', 'probability');

    % Compute the marginal distribution p(ck1)
    ck1Hist = histcounts(ck1, edges, 'Normalization', 'probability');

    % Compute the conditional distribution p(xk2 | xk1)
    condProbs = histVals ./ ck1Hist'; % Normalize each row
    condProbs(isnan(condProbs)) = 0; % Handle NaN values

    % Find the indices of the slices closest to xk1Slice1 and xk1Slice2
    [~, sliceIdx1] = min(abs(edges - ck1Slice1));
    [~, sliceIdx2] = min(abs(edges - ck1Slice2));

    % Plot the conditional probability heatmap
    ax1 = subplot(size(pairs, 1), 2, 1 + (pair-1)*2);
    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0, 0.1]);
    colormap(ax1, 'parula');
    hold on;
    line([edges(1), edges(end)], [edges(sliceIdx1), edges(sliceIdx1)], 'Color', 'yellow', 'LineWidth', 1.5);
    line([edges(1), edges(end)], [edges(sliceIdx2), edges(sliceIdx2)], 'Color', 'red', 'LineWidth', 1.5);
    hold off;
    axis square;
    xlim([-10, 10]);
    ylim([-10, 10]);
    xlabel('c_{k2}');
    ylabel('c_{k1}');
    colorbar;
    title(sprintf("p(c_{k2}|c_{k1}) for k1 = %d , k2 = %d", pairs(pair, 1), pairs(pair, 2)));

    % Plot the slices of xk1
    ax2 = subplot(size(pairs, 1), 2, 2 + (pair-1)*2);
    ck2Hist = histcounts(ck2, edges, 'Normalization', 'probability');
    plot(edges(1:end-1), ck2Hist, 'LineWidth', 1, 'DisplayName', 'p(c_{k2})');
    hold on;
    plot(edges(1:end-1), condProbs(sliceIdx1, :), 'LineWidth', 1, 'DisplayName', sprintf('p(c_{k2} | c_{k1}=%d)', edges(sliceIdx1)));
    plot(edges(1:end-1), condProbs(sliceIdx2, :), 'LineWidth', 1, 'DisplayName', sprintf('p(c_{k2} | c_{k1}=%d)', edges(sliceIdx2)));
    hold off;
    legend('Fontsize',10);
    grid on;
    set(gca, 'GridAlpha', 0.1);
    axis square;
    xlim([-10, 10]);
    ylim([0, 0.165]);
    xlabel('c_{k2}');
    ylabel('Probability Density');
    title(sprintf("Marginal and Conditional at c_{k1} = 1 and c_{k2} = 2"));

end

%% 4. Find Components with largest akj
% Define the array of k values to analyse
k_values = [50, 100];

% Loop over each k value
for idx = 1:length(k_values)
    k = k_values(idx); % Current k value

    % Plot the generative weight of the chosen component
    figure;
    wk = W(:, k); % Generative weight for component k
    imagesc(reshape(wk, [sqrt(1024), sqrt(1024)]), max(abs(wk)) * [-1, 1] + [-1e-5, 1e-5]);
    axis square;
    xlabel(sprintf('W_{%d}', k));
    set(gca, 'yticklabel', '', 'xticklabel', '');
    colormap(gray);
    box on;
    title(sprintf('Generative Weight for Component k = %d', k));

    % Find the ten components with the largest a_{k,j} for the current k
    [~, topComponents] = sort(A(k, :), 'descend');
    topComponents = topComponents(1:10); 

    % Create a figure for plotting top 10 generative weights in one plot
    figure;
    % Loop through top 10 components
    for i = 1:10
        % Create a subplot for each generative weight
        subplot(2, 5, i); % 2 rows, 5 columns for the grid
        wk_top = W(:, topComponents(i)); % Generative weight for top component
        imagesc(reshape(wk_top, [sqrt(1024), sqrt(1024)]), max(abs(wk_top)) * [-1, 1] + [-1e-5, 1e-5]);
        axis square;
        xlabel(sprintf('W_{%d}', topComponents(i)));
        set(gca, 'yticklabel', '', 'xticklabel', '');
        colormap(gray);
        box on;
        title(sprintf('Component %d', topComponents(i)));
    end
    sgtitle(sprintf('Generative Weights of Top 10 Components with Largest a_{%d,j}', k)); % Super title
end

