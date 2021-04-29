clear all;
load digitvars.mat
figure;
h = heatmap(Error);
h.XData = second_layer;
h.YData = first_layer;
h.XLabel = "Second Layer Neurons";
h.YLabel = "First Layer Neurons";
h.Title = "Comparison of different layers in a stacked autoencoder";

j = 1
k = 1
l = 1
Lf = length(first_layer);
Ls = length(second_layer);

for j = 1:Lf
    for k = 1:Ls
        E(Ls*(j-1)+k,:) = Error2(j,k,:);
        Labels(Ls*(j-1)+k) = sprintf("(%d,%d)",first_layer(j),second_layer(k));
    end
end

figure;
plot(E(:,1));
hold on;
plot(E(:,2));
hold on;
plot(E(:,3));
xticklabels(Labels);
xtickangle(45);
legend({"25 Neurons","50 Neurons","75 Neurons"});
xlabel("(Neurons in First Layer, Neurons in Second Layer)");
ylabel("Accuracy");
title("Change in Accuracy for Third Layer");
hold off;

% Layer 1
rng('default')
hiddenSize1 = 100;
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
hiddenSize2 = 100;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

%layer 3

hiddenSize3 = 50;
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