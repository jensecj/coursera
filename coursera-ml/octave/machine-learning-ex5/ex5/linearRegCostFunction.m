function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# calc cost
predictions = X*theta;
errors = (predictions - y);
sq_errors = errors .^ 2;
sum_of_sq_errors = sum(sq_errors);

theta(1) = 0;

 # these two lines are the same, just vectorized
sum_of_sq_theta = theta' * theta;
## sum_of_sq_theta = sum(theta .^ 2);

regularization_term = (lambda/(2*m)) * sum_of_sq_theta;

J = (1/(2*m)) * sum_of_sq_errors + regularization_term;

# calc grad
## errors = predictions .- y; # we've already done this above
par_deriv = 1/m * (errors' * X)';

## theta(1) = 0; % dont regularize theta_0 # done this too
reg_grad_term = (lambda/m) * theta;
grad = par_deriv + reg_grad_term;




% =========================================================================

grad = grad(:);

end
