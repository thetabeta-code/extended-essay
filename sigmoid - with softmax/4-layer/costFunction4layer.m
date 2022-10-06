function [J grad] = costFunction4layer(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_1_size, ...
                                   hidden_layer_2_size, ...
                                   hidden_layer_3_size, ...
                                   hidden_layer_4_size,...
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

Theta1 = reshape(nn_params(1:hidden_layer_1_size * (input_layer_size + 1)), ...
                 hidden_layer_1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_1_size * (input_layer_size + 1))):...
                ((hidden_layer_1_size * (input_layer_size + 1)) + (hidden_layer_2_size * (hidden_layer_1_size + 1)))), ...
                 hidden_layer_2_size, (hidden_layer_1_size + 1));
                 
Theta3 = reshape(nn_params((1 + (hidden_layer_1_size * (input_layer_size + 1)) + (hidden_layer_2_size * (hidden_layer_1_size + 1))):...
                 ((hidden_layer_1_size * (input_layer_size + 1)) + (hidden_layer_2_size * (hidden_layer_1_size + 1)) + hidden_layer_3_size * (hidden_layer_2_size + 1))),...
                 hidden_layer_3_size, (hidden_layer_2_size + 1));
                 
Theta4 = reshape(nn_params((1 + hidden_layer_1_size * (input_layer_size + 1) + hidden_layer_2_size * (hidden_layer_1_size + 1) + hidden_layer_3_size * (hidden_layer_2_size + 1)):...
                 hidden_layer_1_size * (input_layer_size + 1) + hidden_layer_2_size * (hidden_layer_1_size + 1) + hidden_layer_3_size * (hidden_layer_2_size + 1) + hidden_layer_4_size * (hidden_layer_3_size + 1)),...
                 hidden_layer_4_size, (hidden_layer_3_size + 1));
                 
Theta5 = reshape(nn_params((1 + hidden_layer_1_size * (input_layer_size + 1) + hidden_layer_2_size * (hidden_layer_1_size + 1) + hidden_layer_3_size * (hidden_layer_2_size + 1) + hidden_layer_4_size * (hidden_layer_3_size + 1)):end),...
                 num_labels, (hidden_layer_4_size + 1));

m = size(X, 1);
% ======================= Forward propagation =====================================================
y_encoded = (y == (1:num_labels)');

A1 = prependOnes(X);
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
A2 = prependOnes(A2);
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);
A3 = prependOnes(A3);
Z4 = A3 * Theta3';
A4 = sigmoid(Z4);
A4 = prependOnes(A4);
Z5 = A4 * Theta4';
A5 = sigmoid(Z5);
A5 = prependOnes(A5);
Z6 = A5 * Theta5';
A6 = softmax(Z6);

J = -sum(sum(y_encoded' .* log(A6)))/m;
% ================================== Regularization ============================================
Theta_one_square_sum = sum(sum(Theta1(:, 2:end).^2));

Theta_two_square_sum = sum(sum(Theta2(:, 2:end).^2));

Theta_three_square_sum = sum(sum(Theta3(:, 2:end).^2));

Theta_four_square_sum = sum(sum(Theta4(:, 2:end).^2));

Theta_five_square_sum = sum(sum(Theta5(:, 2:end).^2));

regularization = lambda/(2*m) * (Theta_one_square_sum + Theta_two_square_sum + Theta_three_square_sum + Theta_four_square_sum + Theta_five_square_sum);

J = J + regularization;
% ================================== Backpropagation ============================================
DELTA6 = A6 .- y_encoded';
DELTA5 = (DELTA6 * (Theta5(:, 2:end))) .* sigmoidGradient(Z5);
DELTA4 = (DELTA5 * (Theta4(:, 2:end))) .* sigmoidGradient(Z4);
DELTA3 = (DELTA4 * (Theta3(:, 2:end))) .* sigmoidGradient(Z3);
DELTA2 = (DELTA3 * (Theta2(:, 2:end))) .* sigmoidGradient(Z2);

Theta1_grad = (DELTA2' * A1)/m;
Theta2_grad = (DELTA3' * A2)/m;
Theta3_grad = (DELTA4' * A3)/m;
Theta4_grad = (DELTA5' * A4)/m;
Theta5_grad = (DELTA6' * A5)/m;

lambda1_grad = lambda/m * Theta1(:, 2:end);
Theta1_grad(:, 2:end) += lambda1_grad;

lambda2_grad = lambda/m * Theta2(:, 2:end);
Theta2_grad(:, 2:end) += lambda2_grad;

lambda3_grad = lambda/m * Theta3(:, 2:end);
Theta3_grad(:, 2:end) += lambda3_grad;

lambda4_grad = lambda/m * Theta4(:, 2:end);
Theta4_grad(:, 2:end) += lambda4_grad;

lambda5_grad = lambda/m * Theta5(:, 2:end);
Theta5_grad(:, 2:end) += lambda5_grad;
% -------------------------------------------------------------
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:); Theta5_grad(:)];
end
