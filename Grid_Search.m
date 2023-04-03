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

% define the grid of hyperparameters
num_classes = 12;                               % number of classes
num_layers = [2, 3, 4];             % number of layers
num_filters = [64, 128];            % number of filters per layer
filter_size=[6,7];                % number of filters 

% initialise best solution
best_accuracy = 0;
best_params = [];

% auxiliary parameters
aux_params{1} = num_classes;
aux_params{2} = image_size;

% loop over the grid of hyperparameters
for i = 1:length(num_layers)
    for j = 1:length(num_filters)
        for k = 1:length(filter_size)

% current hyperparameters
hyper_params = [num_layers(i), num_filters(j), filter_size(k)];

% create and train model with current hyperparams
layers = create_model(hyper_params,aux_params);

% train model
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
[model,info] = trainNetwork(dsTrain,layers,options);

% extract validation accuracy for current model
accuracy(i,j,k) = info.ValidationAccuracy(end);

% store parameters if they are better than previous
            if accuracy(i,j,k) > best_accuracy
                    best_accuracy = accuracy;
                    best_params = hyper_params;
            end
        end
    end    
end

% display best hyperparameters
disp(['Best hyperparameters: ' num2str(best_params)])
%%
% define a function to create a model
function layers = create_model(hyper_params,aux_params)
% unpack hyperparameter values under test
num_layers = hyper_params(1);
num_filters = hyper_params(2);
filter_size = hyper_params(3);
% unpack auxiliary parameters needed to build network
num_classes = aux_params{1};
image_size = aux_params{2};
% create input layer
layers = [
imageInputLayer(image_size)
];
% create blocks of conv -> batch norm -> relu -> max pool layers
for i = 1:num_layers
layers = [layers
convolution2dLayer(filter_size,num_filters,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)];
end
% output layers
layers = [layers
maxPooling2dLayer(2,'Stride',2)
fullyConnectedLayer(num_classes)
softmaxLayer
classificationLayer];
end