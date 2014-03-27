% Simple demo of AdaBoost with weak MSE classifiers 
% using toy data 'heart_scale' in libSVM package.
%
% Rui Zhao, 2014
% rzhao@ee.cuhk.edu.hk
%

% load customized data [X, Y]: X is nxd matrix and Y is nx1 vector
data_path = 'test_data.mat';
load(data_path);

% function that sample a data from specific distribution
randsample = @(d, n) arrayfun(@(x) find(rand(1)<= cumsum(d(:)), 1), 1:n);

nSamples = size(X, 1);
nBoosts = 400; % number of iterations
sample_rate = 0.1; % set smaller to get weaker classifier

% Initialize data weight vector
W = ones(nSamples, 1)*(1/nSamples);
estimate_cum = zeros(nSamples, 1);
alpha = zeros(nBoosts, 1);
error = zeros(nBoosts, 1);
error_exp = zeros(nBoosts, 1);

for t = 1:nBoosts

   % sample data based on previous weight W_{t-1}
   idx = randsample(W, round(sample_rate*nSamples));
   X_t = X(idx, :);
   Y_t = Y(idx);

   % train weak MSE classifier on X_t
   w_t = pinv(X_t'*X_t)*X_t'*Y_t;

   % compute the error on all data
   estimate = sign(X*w_t);
   err = 0.5 - 0.5*W'*(estimate.*Y);

   % find alpha for this round of AdaBoost
   alpha(t) = 0.5*log((1-err)/max(err,eps));

   % update the data weight
   uW = W.*exp(-alpha(t)*estimate.*Y);
   W = uW/sum(uW);

   % calculate the current error of the cascade of weak classifiers
   estimate_cum = estimate_cum + alpha(t)*estimate;
   error(t) = sum(sign(estimate_cum) ~= Y)/nSamples;
   error_exp(t) = sum(exp(-Y.*estimate_cum))/nSamples;
   
   % plot errors from beginning to current iteration
   plot(error(1:t), 'b', 'LineWidth', 2); hold on;
   plot(error_exp(1:t), 'r', 'LineWidth', 2);
   grid on; xlabel('iteration'); ylabel('error');
   drawnow;

end

