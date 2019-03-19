function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
        #GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha

                                # Initialize some useful values
  m = length(y); # number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

 # ====================== YOUR CODE HERE ======================
# Instructions: Perform a single gradient step on the parameter vector
#               theta.
#
# Hint: While debugging, it can be useful to print out the values
#       of the cost function (computeCostMulti) and gradient here.
#
#
# theta is a vector representing our hypothesis,
# theta_0 * x_0 + ... theta_n * x_n
    predictions = X*theta; # applies our hyp to X by matrix multiplication
    errors = (predictions .- y);
    par_deriv = (1/m) * (errors' * X); ## par_deriv = (1/m) * sum(errors .* X);
    ## par_deriv = (1/m) * (X' * errors); ## par_deriv = (1/m) * sum(errors .* X);
    
    theta = theta - alpha * par_deriv';

    ## delta = ((theta' * X' - y')*X)';
    ## theta = theta - alpha / m * delta;

    ## theta = theta - alpha/m * (X' * (X * theta - y));

        # ============================================================

                                # Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

  end

end
