preprocessing
lag = 50;
        [X,Y] = getTimeSeriesTrainData(Xtrainset,lag);
        [Xpr,Ypr] = getTimeSeriesTrainData(Xpred,lag);
        
        ptr = con2seq(X);
        ttr = con2seq(Y);
        
        
        nn = feedforwardnet(40,'trainlm');
        nn.trainParam.epochs = 50;
        nn = train(nn,ptr,ttr);
       
        Y = sim(nn,con2seq(Xpr));
        err =mse(nn,Ypr,cell2mat(Y));


%{
lag = 50;
[Xtr,Ytr] = getTimeSeriesTrainData(Xtrain,lag); %train on the whole dataset now
[Xpr,Ypr] = getTimeSeriesTrainData(Xpred,lag); %prepare the test set

ptr = con2seq(Xtr);
ttr = con2seq(Ytr);

nn = feedforwardnet(40,'trainlm');
nn.trainParam.epochs = 50;
nn = train(nn,ptr,ttr);

%{
%initializing prediction array based on trainingset data to
%make first predictions
datapredict = [];
datapredict(1,:) = Xtrain(end-lag+1:end,:)';
predictresult = Xtrain(end-lag+1:end,:)';

for i = 1:length(Xpred)
    datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
    ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
    tt = sim(nn, ptest); %Predict the next value
    predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
end

predictpart = predictresult(:,lag+1:end)';
%}
Y = sim(nn,con2seq(Xpr));
Y
err = mse(nn,Ypr,con2seq(Y));
[m,b,r]=postreg(Y,con2seq(Ypr))
%}
        
%fprintf("MSE: %d\nr: %d\n",err,r);
plot(1:250,Xpred(51:300,:),1:250,cell2mat(Y));
xlabel("Data (in Hour)")
ylabel("PM 2.5 Concentration Index")
title("PM 2.5 Value of Shanghai, China in 2017")
legend(["Test Data" "Predicted Data"])

%{
error = zeros(length(Xpred))
for t = 1:length(Xpred);
    error(t) = mse(nn,Xpred(1:t),predictpart(1:t));
end

plot(1:300,error);
xlabel("Time Step");
ylabel("MSE");
We find that the overall error is about ~ 10^3.
%}