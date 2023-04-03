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