clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.


% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;
load('digittrain_dataset');

first_layer = [100,200,300];
second_layer = [50,100,150];
third_layer = [25,50,75];

Error = zeros(3,3);

j = 1;
for hiddenSize1 = first_layer
    k = 1;
    for hiddenSize2 = second_layer
        fprintf("First Layer: %d, Second Layer: %d\n",hiddenSize1,hiddenSize2);
        rng('default')
        % Layer 1
        %hiddenSize1 = 100;
        autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
            'MaxEpochs',400, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',4, ...
            'SparsityProportion',0.15, ...
            'ScaleData', false);
        
        %figure;
        %plotWeights(autoenc1);
        feat1 = encode(autoenc1,xTrainImages);
        
        % Layer 2
        %hiddenSize2 = 100;
        autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
            'MaxEpochs',400, ...
            'L2WeightRegularization',0.002, ...
            'SparsityRegularization',4, ...
            'SparsityProportion',0.1, ...
            'ScaleData', false);
        
        feat2 = encode(autoenc2,feat1);
        
        %layer 3
        %{
hiddenSize3 = 50;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
        'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.01, ...
    'ScaleData', false);

feat3 = encode(autoenc3,feat2);
        %}
        % Layer 3
        softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
        
        
        % Deep Net
        deepnet = stack(autoenc1,autoenc2,softnet);
        
        %cl
        % Test deep net
        imageWidth = 28;
        imageHeight = 28;
        inputSize = imageWidth*imageHeight;
        %[xTestImages, tTest] = digittest_dataset;
        load('digittest_dataset');
        xTest = zeros(inputSize,numel(xTestImages));
        for i = 1:numel(xTestImages)
            xTest(:,i) = xTestImages{i}(:);
        end
        y = deepnet(xTest);
        figure;
        plotconfusion(tTest,y);
        classAcc=100*(1-confusion(tTest,y));
        
        
        % Test fine-tuned deep net
        xTrain = zeros(inputSize,numel(xTrainImages));
        for i = 1:numel(xTrainImages)
            xTrain(:,i) = xTrainImages{i}(:);
        end
        deepnet = train(deepnet,xTrain,tTrain);
        y = deepnet(xTest);
        figure;
        plotconfusion(tTest,y);
        classAcc=100*(1-confusion(tTest,y));
        view(deepnet)
        
        Error(j,k) = classAcc;
        fprintf("Error: %d\n",classAcc);
        k = k + 1;
    end
    j = j + 1;
end


%Now for 3 layers
j = 1;
Error2 = zeros(3,3,3);
for hiddenSize1 = first_layer
    k = 1;
    for hiddenSize2 = second_layer
        l = 1;
        for hiddenSize3 = third_layer
            fprintf("First Layer: %d, Second Layer: %d, Third Layer: %d\n",hiddenSize1,hiddenSize2,hiddenSize3);
            rng('default')
            % Layer 1
            %hiddenSize1 = 100;
            autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
                'MaxEpochs',400, ...
                'L2WeightRegularization',0.004, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.15, ...
                'ScaleData', false);
            
            %figure;
            %plotWeights(autoenc1);
            feat1 = encode(autoenc1,xTrainImages);
            
            % Layer 2
            %hiddenSize2 = 100;
            autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
                'MaxEpochs',400, ...
                'L2WeightRegularization',0.002, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.1, ...
                'ScaleData', false);
            
            feat2 = encode(autoenc2,feat1);
            
            %layer 3
            
            %hiddenSize3 = 50;
            autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
                'MaxEpochs',100, ...
                'L2WeightRegularization',0.004, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.15, ...
                'ScaleData', false);
            
            feat3 = encode(autoenc3,feat2);
            %}
            % Layer 4
            softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);
            
            
            % Deep Net
            deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
            
            %cl
            % Test deep net
            imageWidth = 28;
            imageHeight = 28;
            inputSize = imageWidth*imageHeight;
            %[xTestImages, tTest] = digittest_dataset;
            load('digittest_dataset');
            xTest = zeros(inputSize,numel(xTestImages));
            for i = 1:numel(xTestImages)
                xTest(:,i) = xTestImages{i}(:);
            end
            y = deepnet(xTest);
            figure;
            plotconfusion(tTest,y);
            classAcc=100*(1-confusion(tTest,y));
            
            
            % Test fine-tuned deep net
            xTrain = zeros(inputSize,numel(xTrainImages));
            for i = 1:numel(xTrainImages)
                xTrain(:,i) = xTrainImages{i}(:);
            end
            deepnet = train(deepnet,xTrain,tTrain);
            y = deepnet(xTest);
            figure;
            plotconfusion(tTest,y);
            classAcc=100*(1-confusion(tTest,y));
            view(deepnet)
            
            Error2(j,k,l) = classAcc;
            fprintf("Error: %d\n",classAcc);
            l = l + 1;
        end
        k = k + 1;
    end
    j = j + 1;
end
%Compare with normal neural network (1 hidden layers)
net = patternnet(100);
net=train(net,xTrain,tTrain);
y=net(xTest);
plotconfusion(tTest,y);
classAcc=100*(1-confusion(tTest,y))
view(net)

% %Compare with normal neural network (2 hidden layers)
%
net2 = patternnet([100,100]);
net2 = train(net2,xTrain,tTrain);
y2 = net(xTest);
plotconfusion(tTest,y2);
classAcc = 100*(1-confusion(tTest,y2))
view(net)