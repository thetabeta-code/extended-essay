function [] = ex(num_layers, hidden_layer_size)

%% Initialization

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

maxIter = 400;

load("MNISTtestX.mat");
load("MNISTtestY.mat");
load("MNISTtrainX.mat");
load("MNISTtrainY.mat");

if(num_layers == 1)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/1-layer");
  for(iteration = 1:5)
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/1-layer");
    ex1layer(trainX, trainY, testX, testY, hidden_layer_size, iteration, maxIter);
  endfor
elseif(num_layers == 2)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/2-layer");
  for(iteration = 1:5)
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/2-layer");
    ex2layer(trainX, trainY, testX, testY, hidden_layer_size, iteration, maxIter);
  endfor
elseif(num_layers == 3)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/3-layer");
  for(iteration = 1:5)
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/3-layer");
    ex3layer(trainX, trainY, testX, testY, hidden_layer_size, iteration, maxIter);
  endfor
elseif(num_layers == 4)
  addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/4-layer");
  for(iteration = 1:5)
    cd "D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax";
    addpath("D:/Oakridge/DP-1/Extended_essay/Dataset/ReLU - with softmax/4-layer");
    ex4layer(trainX, trainY, testX, testY, hidden_layer_size, iteration, maxIter);
  endfor
endif

endfunction