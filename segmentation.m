% By Ngoc Hoa Nguyen (20606285)
close all;
clear;

%% Paths & Parameters
imageFolder = fullfile('daffodilSeg', 'ImagesRsz256');
labelFolder = fullfile('daffodilSeg', 'LabelsRsz256');

inputSize = [256 256 3];
classes = ["flower", "background"];
labelIDs = [1, 3];

%% Create Datastores
imds = imageDatastore(imageFolder);

pxds = pixelLabelDatastore(labelFolder, classes, labelIDs, ...
    'ReadFcn', @customLabelReader);

% Split 80% training, 20% validation
% Get number of images
numImages = numel(imds.Files);

% 80/20 - Indices
numTrain = round(0.8 * numImages); % 80% train
trainIdx = 1:numTrain; % 0 -> 80
valIdx   = numTrain+1:numImages; % 80 -> 100

% Split image and label Datastore
imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);

imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);

% Combine into final datasets
train_data = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
val_data   = pixelLabelImageDatastore(imdsVal, pxdsVal);
%% Load Pretrained Network
pre_net = deeplabv3plusLayers(inputSize, 2, 'resnet18');

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', val_data, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% Train the Network
fprintf("Segmentation network is being trained.\n");
segmentnet = trainNetwork(train_data, pre_net, options);

%% Save Trained Model
save('segmentnet.mat', 'segmentnet');

% View the Network Architecture in MATLAB’s Deep Network Designer.
deepNetworkDesigner(segmentnet)

%% Evaluate 

% Do segmentation, save output images to disk
pxdsResults = semanticseg(imds,segmentnet);

I = readimage(imds, 52);
C = semanticseg(I, segmentnet);

% Visualize prediction
B = labeloverlay(I, C);
figure; 
imshow(B); 
title("Predicted Segmentation");

% Evaluate on validation set
metrics = evaluateSemanticSegmentation(pxdsResults, pxds);

dataset_metrics = metrics.DataSetMetrics
class_metrics = metrics.ClassMetrics
normalized_confusion_matrix = metrics.NormalizedConfusionMatrix

% Show Confusion Matrix
figure;
confusionchart(metrics.ConfusionMatrix.Variables, classes, Normalization='row-normalized')

% mean IOU
meanIOU = metrics.ImageMetrics.MeanIoU;

% make histogram of IOU values
histogram(meanIOU)

%% Helper function
function label = customLabelReader(filename)
    ori_label = imread(filename);
    if ~ori_label == 1
        ori_label = 3;
    end
    label = uint8(ori_label);

end