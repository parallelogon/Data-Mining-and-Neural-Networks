load('lasertrain.dat');
load('laserpred.dat');

rng("shuffle");
%we do the same for an unoptomized net with neurons = 10 and lag = 16
untunedLag = 16;%8;
[Xtr,Ytr] = getTimeSeriesTrainData(lasertrain,untunedLag);

ptr = con2seq(Xtr);
ttr = con2seq(Ytr);


%Make and train the NN
nn2 = feedforwardnet([10,3],'trainlm');
nn2.trainParam.epochs = 50;
nn2=train(nn2,ptr,ttr);

%The aim is to use our iterative procedure to predict the next values
%initializing prediction array
datapredict = [];
datapredict(1,:) = lasertrain(end-untunedLag+1:end,:)';
predictresult = lasertrain(end-untunedLag+1:end,:)';

%Loops through adding the predicted result onto the data used
%to predict for 100 total new predictions
for i = 1:100
    datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
    ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
    tt = sim(nn2, ptest); %Predict the next value
    predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
end
predictpartUnTuned = predictresult(:,untunedLag+1:end)';
ErrorUnTuned = mse(predictpartUnTuned,laserpred);

xes = 1:100;
plot(xes,laserpred,xes,predictpartUnTuned);
xlabel("Time Step");
ylabel("Intensity");
titletext = sprintf("MLP With Two Layers and MSE: %d",ErrorUnTuned);
title(titletext);
