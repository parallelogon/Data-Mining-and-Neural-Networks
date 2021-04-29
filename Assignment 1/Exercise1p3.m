%Create a dataset with some noise
N = 75;
x = linspace(0,3*pi,N);
Lx = length(x);
y = sin(x.^2) + randn(1,Lx);

%Pick out pieces at random to make a validation and a test set.
picker = randn(1,Lx);

%15 percent become test set
Tpicks = picker >= 0.85;
XTest = x(Tpicks);
YTest = y(Tpicks);

%15 percent become validation set
valpicks = 0.85 > picker >= 0.70;
XVal = x(valpicks);
YVal = y(valpicks);

%Leftovers become part of the training set.
leaves = picker < 0.70;
XTrain = x(leaves);
YTrain = y(leaves);

%Set hyperparameter depth of NN.
depth = 2.^(1:6);

%Initialize an array for the MSE of different architectures
MSE = zeros(2,length(depth));
idb = 1; %id of bayesian network
for i = depth
    bayesnet = feedforwardnet(i,'trainbr');
    bayesnet = train(bayesnet,XTrain,YTrain);
    Y = sim(bayesnet,XVal);
    Id = find(depth == i);
    TMSE = mse(bayesnet,Y,YVal);
    
    %Checks if this is the lowest MSE, if so keeps the classifier for later
    if TMSE < min(MSE(1,MSE(1,:)~=0));
        BestBayes = bayesnet;
        idb = i
    end
    MSE(1,Id) = TMSE;
end

idnb = 1;
for i = depth;
    %Training
    nonbayesnet = feedforwardnet(i);
    nonbayesnet = train(nonbayesnet,XTrain,YTrain);
    
    %Validating
    Ynb = sim(nonbayesnet,XVal);
    Id = find(depth == i);
    TMSE = mse(nonbayesnet,Ynb,YVal);
    
    %Checks if this is the lowest MSE, if so keeps the classifier for later
    if TMSE < min(MSE(2,MSE(2,:)~=0));
        BestNonBayes = nonbayesnet;
        fprintf("Number of Neurons NonBayes: %d",i);
        idnb = i
    end
    MSE(2,Id) = TMSE;
end


subplot(2,1,1);
plot(x,y,x,sim(BestBayes,x));
xlabel("x");
ylabel("y");
titleTextBayes = sprintf("Best Bayesian Network with %d Neurons",idb);
title(titleTextBayes);
subplot(2,1,2);
plot(x,y,x,sim(BestNonBayes,x));
xlabel("x");
ylabel("y");
titleTextNonBayes = sprintf("Best Non-Bayesian Network with %d Neurons",idnb);
title(titleTextNonBayes);
fprintf("MSE for best bayesian NN: %d\nMSE for best nonbayesian NN: %d",mse(BestBayes,sim(BestBayes,XTest),YTest),mse(BestNonBayes,sim(BestNonBayes,XTest),YTest))
