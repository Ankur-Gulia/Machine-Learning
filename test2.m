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
% define constant parameters
num_classes = 12;  % number of classes
num_filters = 32;  % base number of filters in convolutional layers
filter_size = 7;  % convolutional filter size

% define network layers
layers = [
    imageInputLayer([image_size 1])
    
    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer

    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer

    reluLayer
    
    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(filter_size,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)

    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];
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
net = trainNetwork(dsTrain,layers,options);
% classify the validation output using the trained network
[YPred,probs] = classify(net,dsVal);
% extract ground truth labels
YVal = imdsVal.Labels;
% accuracy in percent
accuracy = 100*sum(YPred == YVal)/numel(YVal);
disp(['The accuracy is: ' num2str(accuracy)])

% plot confusion matrix
figure;
plotconfusion(YVal,YPred)
% Display sample test images with predicted labels and
% the predicted probabilities of the images having those labels.
idx = randperm(numel(imdsVal.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsVal,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label)+ ", "+num2str(100*max(probs(idx(i),:)),3)+"%");
end
disp(["Validation Set Accuracy: " num2str(accuracy) "%"]);