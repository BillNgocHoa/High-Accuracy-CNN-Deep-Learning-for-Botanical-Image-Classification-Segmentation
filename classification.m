% By Ngoc Hoa Nguyen

close all;
clear; 
clc;

%% Parameters
inputSize = [256 256 3];
numClasses = 17;
dataFolder = fullfile('17flowers');

%% Load Dataset
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split 80/20 for training/validating
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% Define Augmentation & Preprocessing
augmenter = imageDataAugmenter( ...
    'RandRotation', [-40, 40], ...
    'RandXTranslation', [-0.2, 0.2], ...
    'RandYTranslation', [-0.2, 0.2], ...
    'RandXShear', [-20, 20], ...
    'RandYShear', [-20, 20], ...
    'RandXScale', [0.8, 1.2], ...
    'RandYScale', [0.8, 1.2], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter);
augVal = augmentedImageDatastore(inputSize, imdsVal); % resize the Val data


%% CNN Architecture
relu_maxpool = [
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    ];

conv_layers = [
    convolution2dLayer(3, 32, 'Padding', 'same')
    relu_maxpool

    convolution2dLayer(3, 64, 'Padding', 'same')
    relu_maxpool

    convolution2dLayer(3, 128, 'Padding', 'same')
    relu_maxpool

    convolution2dLayer(3, 256, 'Padding', 'same')
    relu_maxpool
    ];

fully_connect_layer = [
    fullyConnectedLayer(512)
    reluLayer()
    ];

final_layer = [
    fullyConnectedLayer(numClasses)
    softmaxLayer()
    classificationLayer()
    ];

layers = [
    imageInputLayer(inputSize)

    conv_layers

    dropoutLayer(0.3)

    fully_connect_layer

    dropoutLayer(0.5)

    final_layer
];

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train Network
fprintf("Classfication network is being trained.\n");
classnet = trainNetwork(augTrain, layers, options);

%% Save Trained Model
save('classnet.mat', 'classnet');

% View the Network Architecture in MATLAB’s Deep Network Designer.
deepNetworkDesigner(classnet)