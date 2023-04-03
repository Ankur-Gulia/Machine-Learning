clear

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Data 

% create an image data store from the raw images 
imdsTrain = imageDatastore('speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames");

% create an image validation data store from the validation images 
imdsVal = imageDatastore('speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames");

%%
% use the transform function to resize each image
image_size = [98 50];
dsTrain = augmentedImageDatastore(image_size,imdsTrain,'ColorPreprocessing', 'none');
dsVal = augmentedImageDatastore(image_size,imdsVal,'ColorPreprocessing','none');
%%
%Training Set
shuffled_idx1 = randperm(dsTrain.NumObservations);
shuffled_idx2 = randperm(dsTrain.NumObservations);
shuffled_idx3 = randperm(dsTrain.NumObservations);
subset_size1 = floor(dsTrain.NumObservations);
subset_size2 = floor(dsTrain.NumObservations);
subset_size3 = floor(dsTrain.NumObservations);


dsTrainsubset1 = subset(dsTrain, shuffled_idx1(1:subset_size1));
dsTrainsubset2 = subset(dsTrain, shuffled_idx2(1:subset_size2));
dsTrainsubset3 = subset(dsTrain, shuffled_idx3(1:subset_size3));

% Validation Set
shuffled_idx4 = randperm(dsVal.NumObservations);
shuffled_idx5 = randperm(dsVal.NumObservations);
shuffled_idx6 = randperm(dsVal.NumObservations);
subset_size4 = floor(dsVal.NumObservations);
subset_size5 = floor(dsVal.NumObservations);
subset_size6 = floor(dsVal.NumObservations);

dsValsubset1 = subset(dsVal, shuffled_idx4(1:subset_size4));
dsValsubset2 = subset(dsVal, shuffled_idx5(1:subset_size5));
dsValsubset3 = subset(dsVal, shuffled_idx6(1:subset_size6));
%% model 1
% define constant parameters
num_classes = 12;  % number of classes
num_filters = 32;  % base number of filters in convolutional layers
filter_size = 7;  % convolutional filter size
%%
% define network layers
layers = [
    imageInputLayer([image_size 1])
    
    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(filter_size,2*num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(filter_size,2*num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.2)

    maxPooling2dLayer([12,1])
    
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];
%%
% display network design
analyzeNetwork(layers)
% training options 
options = trainingOptions('adam', ...
    "MiniBatchSize",20, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',10, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','cpu');

% train network
net = trainNetwork(dsTrainsubset1,layers,options);
%% model 2
% define constant parameters
num_classes2 = 12;  % number of classes
num_filters2 = 64;  % base number of filters in convolutional layers
filter_size2 = 6;  % convolutional filter size
%%
% define network layers
layers2 = [
    imageInputLayer([image_size 1])
    
    convolution2dLayer(filter_size2,num_filters2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(filter_size2,num_filters2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(filter_size2,2*num_filters2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(filter_size2,2*num_filters2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.2)

    maxPooling2dLayer([12,1])
    
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];
%%
% display network design
analyzeNetwork(layers2)
% training options 
options_2 = trainingOptions('adam', ...
    "MiniBatchSize",20, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',10, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','cpu');

% train network
net_2 = trainNetwork(dsTrainsubset2,layers2,options_2);
%% model 3
% define constant parameters
num_classes = 12;  % number of classes
num_filters3 = 32;  % base number of filters in convolutional layers
filter_size3 = 7;  % convolutional filter size

% define network layers
layers3 = [
    imageInputLayer([image_size 1])
    
    convolution2dLayer(7,num_filters3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(7,num_filters3,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(7,num_filters3,'Padding','same')
    batchNormalizationLayer
    reluLayer


    convolution2dLayer(7,num_filters3,'Padding','same')
    batchNormalizationLayer
    reluLayer

    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(7,num_filters3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];
% display network design
analyzeNetwork(layers3)
% training options 
options3 = trainingOptions('adam', ...
    "MiniBatchSize",20, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',10, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','cpu');

% train network
net_3 = trainNetwork(dsTrainsubset3,layers3,options3);
%% Predicitions
% classify the validation output using the trained network 1
YPred1= classify(net,dsVal);
% extract ground truth labels
YVal = imdsVal.Labels;

% accuracy in percent
accuracy1 = 100*sum(YPred1 == YVal)/numel(YVal);
% classify the validation output using the trained network
YPred2 = classify(net_2,dsVal);
% extract ground truth labels
YVal = imdsVal.Labels;

% accuracy in percent
accuracy2 = 100*sum(YPred2 == YVal)/numel(YVal);
% classify the validation output using the trained network
YPred3 = classify(net_3,dsVal);
% extract ground truth labels
YVal = imdsVal.Labels;

% accuracy in percent
accuracy3 = 100*sum(YPred3 == YVal)/numel(YVal);

%Ensemble pred 
ensemblePred = mode([YPred1, YPred2, YPred3],2);
ensembleaccuracy = 100*sum(ensemblePred == YVal)/numel(YVal);
disp(["Validation Set Accuracy1: " num2str(accuracy1) "%"]);
disp(["Validation Set Accuracy2: " num2str(accuracy2) "%"]);
disp(["Validation Set Accuracy3: " num2str(accuracy3) "%"]);
disp(["Validation Set Ensemble Accuracy: " num2str(ensembleaccuracy) "%"]);