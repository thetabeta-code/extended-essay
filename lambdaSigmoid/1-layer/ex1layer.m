function cv_accuracy = ex1layer(trainX, trainY, cvX, cvY, ...
                                initial_nn_params, hidden_layer_size, lambda, maxIter)

%% Setup the parameters
input_layer_size  = 784;
hidden_layer_1_size = hidden_layer_size;
num_labels = 10;

cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/1-layer";

m = size(trainX, 1);

fprintf('Lambda = %f', lambda);

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', maxIter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costFunction1layer(p, ...
                                   input_layer_size, ...
                                   hidden_layer_1_size, ...
                                   num_labels, trainX, trainY, lambda);
                                   
first_cost = costFunction1layer(
                            initial_nn_params,...
                            input_layer_size,...
                            hidden_layer_1_size,...
                            num_labels,...
                            trainX, trainY, lambda);     

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

[nn_params, cost, i, costvals] = fmincg(costFunction, initial_nn_params, options);

% Obtain Thetas back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_1_size * (input_layer_size + 1)), ...
                 hidden_layer_1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_1_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_1_size + 1));

newcostvals = [first_cost; costvals];

pred = predict1layer(Theta1, Theta2, trainX);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == trainY')) * 100);

final_accuracy = mean(double(pred == trainY')) * 100;

predCV = predict1layer(Theta1, Theta2, cvX);

cv_accuracy = mean(double(predCV == cvY')) * 100;

fprintf('\nCross validation Set Accuracy: %f\n', cv_accuracy);

cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid";
end