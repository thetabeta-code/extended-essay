function [] = ex(num_layers, hidden_layer_size)

lambVec = [0.08 0.1 1.3 1.5 1.7 2];
accVec = zeros(length(lambVec));
bestAcc = 0;
num_labels = 10;
input_layer_size = 784;
maxIter = 400;

load("MNISTcvX.mat");
load("MNISTcvY.mat");
load("MNISTtrainX.mat");
load("MNISTtrainY.mat");

if(num_layers == 1)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/1-layer");
  initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

  % Unroll parameters
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
  for(lambIter = 1:length(lambVec))
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/1-layer");
    accVec(lambIter)= ex1layer(trainX, trainY, cvX, cvY, initial_nn_params, hidden_layer_size, lambVec(lambIter), maxIter);
    if(accVec(lambIter) > bestAcc)
      bestLambda = lambIter;
      bestAcc = accVec(lambIter);
    endif
  endfor
  fprintf("Best lambda for %f layers %f hidden neurons = %f\n", num_layers, hidden_layer_size, lambVec(bestLambda));
elseif(num_layers == 2)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/2-layer");
  initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

  % Unroll parameters
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];
  for(lambIter = 1:length(lambVec))
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/2-layer");
    accVec(lambIter)= ex2layer(trainX, trainY, cvX, cvY, initial_nn_params, hidden_layer_size, lambVec(lambIter), maxIter);
    if(accVec(lambIter) > bestAcc)
      bestLambda = lambIter;
      bestAcc = accVec(lambIter);
    endif
  endfor
  fprintf("Best lambda for %f layers %f hidden neurons = %f\n", num_layers, hidden_layer_size, lambVec(bestLambda));
elseif(num_layers == 3)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/3-layer");
  initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta4 = randInitializeWeights(hidden_layer_size, num_labels);

  % Unroll parameters
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:); initial_Theta4(:)];
  for(lambIter = 1:length(lambVec))
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/3-layer");
    accVec(lambIter)= ex3layer(trainX, trainY, cvX, cvY, initial_nn_params, hidden_layer_size, lambVec(lambIter), maxIter);
    if(accVec(lambIter) > bestAcc)
      bestLambda = lambIter;
      bestAcc = accVec(lambIter);
    endif
  endfor
  fprintf("Best lambda for %f layers %f hidden neurons = %f\n", num_layers, hidden_layer_size, lambVec(bestLambda));
elseif(num_layers == 4)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/4-layer");
  initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta4 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
  initial_Theta5 = randInitializeWeights(hidden_layer_size, num_labels);

  % Unroll parameters
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:); initial_Theta4(:); initial_Theta5(:)];
  for(lambIter = 1:length(lambVec))
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/lambdaSigmoid/4-layer");
    accVec(lambIter)= ex4layer(trainX, trainY, cvX, cvY, initial_nn_params, hidden_layer_size, lambVec(lambIter), maxIter);
    if(accVec(lambIter) > bestAcc)
      bestLambda = lambIter;
      bestAcc = accVec(lambIter);
    endif
  endfor
  fprintf("Best lambda for %f layers %f hidden neurons = %f\n", num_layers, hidden_layer_size, lambVec(bestLambda));
endif

output = lambVec(bestLambda);

save(strcat("sigmoid_L", int2str(num_layers), "N", int2str(hidden_layer_size), "I", int2str(maxIter), "bestlambda.dat"), 'output');

endfunction