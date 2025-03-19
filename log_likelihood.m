function [L, grad] = log_likelihood(params, X)
    % Compute the log-likelihood and its gradient for the conditional model
    %
    % Inputs:
    %   params - Flattened parameter vector (logA and logb)
    %   X      - Data matrix of latent variables (N x K)
    %
    % Outputs:
    %   L    - Negative log-likelihood 
    %   grad - Gradient of the negative log-likelihood w.r.t. log parameters

    [N, K] = size(X);

    % Extract parameters from input vector
    logA = reshape(params(1:K*K), K, K);  % Log-transformed A matrix
    logb = params(K*K+1:end);             % Log-transformed b vector

    A = exp(logA);
    b = exp(logb);

    % Enforce no self-dependencies (set diagonals to zero)
    A(logical(eye(K))) = 0;

    % Initialise log-likelihood and gradients
    L = 0;
    grad_A = zeros(K, K);
    grad_b = zeros(K, 1);
    
    %Calculate sigma_squared
    sigma2 = X.^2 * A + repmat(b', N, 1);

    %Compute loss
    L = sum(0.5 * log(2 * pi * sigma2) + (X.^2) ./ (2 * sigma2), 'all');

    %Compute gradient
    dLdsigma2 = (X.^2 - sigma2) ./ (2 * sigma2.^2);
    grad_log_A = -(X.^2)' * dLdsigma2 .* A;
    grad_log_b = -sum(dLdsigma2, 1)' .* b;

    % Vectorise
    grad = [grad_log_A(:); grad_log_b];
end
