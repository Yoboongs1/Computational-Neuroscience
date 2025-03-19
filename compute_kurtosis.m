function k = compute_kurtosis(data)
    % Compute the excess kurtosis of a dataset
    mu = mean(data); % Mean of the data
    sigma = std(data); % Standard deviation of the data
    mu4 = mean((data - mu).^4); % Fourth central moment
    k = (mu4 / sigma^4) - 3; % Excess kurtosis
end