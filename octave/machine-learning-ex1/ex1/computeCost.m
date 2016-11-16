function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% pseudocode:
% J(theta) = (1/2*m) * ( sum_1^m(theta[0] + theta[1]*x[i] - y[i]) )^ 2

% straight conversion from pseudocode:
% we're doing linear regression in 2d, so we only have two values for theta

## t0 = theta(1);
## t1 = theta(2);
## for i = 1:m
##   prediction = t0 + t1*X(i,2); # we can discard the first 'only-1's'
##                                # column when doing this the summation way
##   error = prediction - y(i);
##   errors(i) = error ^ 2;
## end

## sum_of_sq_errors = sum(errors);
## J = (1/(2*m)) * sum_of_sq_errors;


% or we could do it by vectorization:
% we only have 1 feature, and we added a new column in front of the
% first column of X, consisting only on 1's, so X is a m*2 matrix and
% y is a 2*1 vector, because we're in 2d, so we can easily multiply them.

predictions = X*theta;
errors = (predictions - y);
sq_errors = errors .^ 2;
sum_of_sq_errors = sum(sq_errors);

J = (1/(2*m)) * sum_of_sq_errors;

% =========================================================================
% this is the same as above, the hyp is constructed as in predictions
% because our first 'features' (the row we added to X) is a vector of ones
## M = X(:,2);
## predictions = theta(1) + theta(2)*M;
## errors = (predictions - y);
## sq_errors = errors .^ 2;
## sum_of_sq_errors = sum(sq_errors);

## J = (1/(2*m)) * sum_of_sq_errors;




end
