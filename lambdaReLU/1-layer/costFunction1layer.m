function [J grad] = costFunction1layer(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_1_size, ...
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

Theta1 = reshape(nn_params(1:hidden_layer_1_size * (input_layer_size + 1)), ...
                 hidden_layer_1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_1_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_1_size + 1));
                 
m = size(X, 1);
% ======================= Forward propagation =====================================================

y_encoded = (y == (1:num_labels)');

A1 = prependOnes(X);
Z2 = A1 * Theta1';
A2 = ReLU(Z2);
A2 = prependOnes(A2);
Z3 = A2 * Theta2';
A3 = softmax(Z3);

J = -sum(sum(y_encoded' .* log(A3)))/m;

% ================================== Regularization ============================================

Theta_one_square_sum = sum(sum(Theta1(:, 2:end).^2));

Theta_two_square_sum = sum(sum(Theta2(:, 2:end).^2));

regularization = lambda/(2*m) * (Theta_one_square_sum + Theta_two_square_sum);

J = J + regularization;

% ================================== Backpropagation ============================================
DELTA3 = A3 - y_encoded';
DELTA2 = (DELTA3 * (Theta2(:, 2:end))) .* ReLUGradient(Z2);

Theta1_grad = (DELTA2' * A1)/m;
Theta2_grad = (DELTA3' * A2)/m;

lambda1_grad = lambda/m * Theta1(:, 2:end);
Theta1_grad(:, 2:end) += lambda1_grad;

lambda2_grad = lambda/m * Theta2(:, 2:end);
Theta2_grad(:, 2:end) += lambda2_grad;

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end