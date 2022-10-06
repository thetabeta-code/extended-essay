function [] = ex1layer(trainX, trainY, testX, testY,...
                  hidden_layer_size, iteration, maxIter)

%% Setup the parameters
input_layer_size  = 784;
hidden_layer_1_size = hidden_layer_size;   
num_labels = 10;

cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/1-layer";

m = size(trainX, 1);

fprintf('Run %f', iteration);

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_1_size);
initial_Theta2 = randInitializeWeights(hidden_layer_1_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

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

%  You should also try different values of lambda
lambda = 0.03;

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
                 
%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

newcostvals = [first_cost; costvals];

fprintf('\nProgram paused. Press enter to continue.\n');


%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict1layer(Theta1, Theta2, trainX);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == trainY')) * 100);

final_accuracy = mean(double(pred == trainY')) * 100;

predTest = predict1layer(Theta1, Theta2, testX);

test_accuracy = mean(double(predTest == testY')) * 100;

fprintf('\nTest Set Accuracy: %f\n', test_accuracy);

csvwrite(strcat('ReLU_L1N', int2str(hidden_layer_size), 'I', int2str(maxIter), '_R', int2str(iteration), '.dat'), newcostvals);
csvwrite(strcat('ReLU_L1N', int2str(hidden_layer_size), 'I', int2str(maxIter), '_R', int2str(iteration), '_acc.dat'), final_accuracy);
csvwrite(strcat('ReLU_L1N', int2str(hidden_layer_size), 'I', int2str(maxIter), '_R', int2str(iteration), '_testacc.dat'), test_accuracy);

save(strcat('ReLU_L1N', int2str(hidden_layer_size), 'I', int2str(maxIter), '_R', int2str(iteration), 'Theta1.mat'), 'Theta1');
save(strcat('ReLU_L1N', int2str(hidden_layer_size), 'I', int2str(maxIter), '_R', int2str(iteration), 'Theta2.mat'), 'Theta2');

cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax";
end