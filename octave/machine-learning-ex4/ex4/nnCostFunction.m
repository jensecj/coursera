function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

% Setup some useful variables
  m = size(X, 1);

% You need to return the following variables correctly
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
#
#
#
#

% make a matrix out of our expected outputs,
# allows us to easily calculate using matrix multiplication
  y_matrix = eye(num_labels)(y,:);

# FORWARD PROPAGATION
  a1 = [ones(m, 1) X];

  z2 = a1 * Theta1';
  a2 = [ones(m, 1) sigmoid(z2)];

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

% FP - compute the cost
# this is almost completely cloned from ex3,
# except that now y's a matrix, so we use element-wise multiplication,
# and sum twice to sum through the entire result matrices.
  hypothesis = a3;

  first_term = -y_matrix .* log(hypothesis);
  second_term = (1-y_matrix) .* log(1-hypothesis);

  unreg_cost = (1/m) * (first_term - second_term);
  unreg_cost = sum(sum(unreg_cost));

# FP regularization, also cloned from ex3, but now we have 2 thetas
  no_bias_theta1 = Theta1(:, 2:end);% exclude the bias units (first column)
  no_bias_theta2 = Theta2(:, 2:end);% exclude the bias units (first column)

  sq_err_theta1 = sum(sum(no_bias_theta1 .^ 2));
  sq_err_theta2 = sum(sum(no_bias_theta2 .^ 2));

  sq_err_thetas = sq_err_theta1 + sq_err_theta2;

  reg_term = (lambda/(2*m)) * sq_err_thetas;

  J = unreg_cost + reg_term;

# BACK PROPAGATION
  d3 = a3 - y_matrix;% error for output neurons

  d2 = (d3 * no_bias_theta2) .* sigmoidGradient(z2);% errors for 1 layer behind output (hidden layers)

  D1 = d2' * a1; # Gradient (error-rate * activation), i guess this is how confident we are in out guess
  D2 = d3' * a2; # Gradient (error-rate * activation)

  Theta1_grad = (1/m) * D1; % average out our predictions, so we minimize our average error
  Theta2_grad = (1/m) * D2;

# BP regularization
  Theta1(:,1) = 0;# do not regularize theta_0
  Theta2(:,1) = 0;# do not regularize theta_0

  reg_term_1 = (lambda/m) * Theta1; # control how much we regularize our output
  reg_term_2 = (lambda/m) * Theta2;

  Theta1_grad = Theta1_grad + reg_term_1;
  Theta2_grad = Theta2_grad + reg_term_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
