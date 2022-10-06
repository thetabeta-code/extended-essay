function p = predict4layer(Theta1, Theta2, Theta3, Theta4, Theta5, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = ReLU([ones(m, 1) X] * Theta1');
h2 = ReLU([ones(m, 1) h1] * Theta2');
h3 = ReLU([ones(m, 1) h2] * Theta3');
h4 = ReLU([ones(m, 1) h3] * Theta4');
h5 = softmax([ones(m, 1) h4] * Theta5');
[dummy, p] = max(h5, [], 2);

% =========================================================================


end
