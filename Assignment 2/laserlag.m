load('lasertrain.dat');
load('laserpred.dat');

%Generates a list of lags and neurons with increments of delta given by
%(b-a)/delta + 1 = #of increments
laglist = linspace(8,88,(88-8)/8+1); %8-80 ~ten periods of 8, the periodicity of the laser peaks
neuronlist = linspace(10,60,(60-10)/10+1); %10-90 with delta = 10.  testing the amount of effective overparametrization

%Initialize an error matrix for each of the values of neurons/lags
Error = zeros(length(neuronlist),length(laglist));

%loop through number of neurons in hidden layer and lagings.  Since this
%can be succeptible to variations in the training process, average it.
lagindex = 1;
neuronindex = 1;
iterations = 1;
for it = 1:iterations;
    for neuron = neuronlist;
        lagindex = 1;
        for lag = laglist;
            trainingset = lasertrain(1:700);
            valiset = lasertrain(700:end);
            
            %Training set
            [Xtr,Ytr] = getTimeSeriesTrainData(trainingset, lag);
            [Xvali,Yvali] = getTimeSeriesTrainData(valiset,lag);
            
            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);
            
            %creation of networks
            net=feedforwardnet(neuron,'trainlm');
            
            %training and simulation with epochs being 50 for timing
            net.trainParam.epochs = 50;
            net=train(net,ptr,ttr);
            
            Y = sim(net,con2seq(Xvali))
            %{
            %initializing prediction array based on trainingset data to
            %make first predictions
            datapredict = [];
            datapredict(1,:) = trainingset(end-lag+1:end,:)';
            predictresult = trainingset(end-lag+1:end,:)';
            
       
            %Loops through adding the predicted result onto the data used
            %to predict for 100 total new predictions
            for i = 1:300
                datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
                ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
                tt = sim(net, ptest); %Predict the next value
                predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
            end
            
            %Isolate the predicted part of the dataset and compare it to
            %the validation set to compute the error.
            predictpart = predictresult(:,lag+1:end)';
            %}
            
            Error(neuronindex,lagindex) = Error(neuronindex,lagindex) + mse(net,valiset,cell2mat(Y))
            
            lagindex = lagindex + 1;
        end
        neuronindex = neuronindex + 1;
    end
end

%average the total error for all iterations
Error = Error / iterations;

%Generate a heatmap to look at the final errors by neuron and lagging
h = heatmap(Error);
h.Title = "MSE for combinations of depth and lagging";
h.XLabel = "Lag";
h.YLabel = "Number of Neurons";
h.XData = laglist;
h.YData = neuronlist;

%find row column of minimum error measurement
min = max(Error(1,:)); %initialize min as max of one row of Error matrix
loc = [0 0]
for i = 1:length(Error(:,1)),
    for j = 1:length(Error(1,:)),
        if Error(i,j) < min,
            min = Error(i,j);
            loc = [i j];
        end
    end
end

tunedNeurons = loc(1)
tunedLag = loc(2)
fprintf("Minimum Error is %d with %d Neurons and a lag of %d\n",min,neuronlist(tunedNeurons),laglist(tunedLag));


h = heatmap(Error);
h.Title = "MSE for combinations of depth and lagging";
h.XLabel = "Lag";
h.YLabel = "Number of Neurons";
h.XData = laglist;
h.YData = neuronlist;

%Comparing a NN with tunedNeurons and tunedLag with a NN with 50 neurons
%and a lag of 80(one period).
lag = 80%laglist(tunedLag)
[Xtr,Ytr] = getTimeSeriesTrainData(lasertrain,lag);

ptr = con2seq(Xtr);
ttr = con2seq(Ytr);

%Make and train the NN
net1 = feedforwardnet(50,'trainlm');%neuronlist(tunedNeurons),'trainlm');
net1=train(net1,ptr,ttr);

%The aim is to use our iterative procedure to predict the next values
%initializing prediction array
datapredict = [];
datapredict(1,:) = lasertrain(end-lag+1:end,:)';
predictresult = lasertrain(end-lag+1:end,:)';

%Loops through adding the predicted result onto the data used
%to predict for 100 total new predictions
for i = 1:100
    datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
    ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
    tt = sim(net1, ptest); %Predict the next value
    predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
end

predictpartTuned = predictresult(:,lag+1:end)';
ErrorTuned = mse(predictpartTuned,laserpred);
fprintf("Error of tuned NN: %d\n",ErrorTuned);


%we do the same for an unoptomized net with neurons = 10 and lag = 8
untunedLag = 8
[Xtr,Ytr] = getTimeSeriesTrainData(lasertrain,untunedLag);

ptr = con2seq(Xtr);
ttr = con2seq(Ytr);

%Make and train the NN
nn2 = feedforwardnet(10,'trainlm')
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

fprintf("Error of untuned NN: %d\n",ErrorUnTuned);
fprintf("Error of tuned NN: %d\n",ErrorTuned);
%Finally graphs of the tuned NN and the untuned NN versus the correct
%values
subplot(2,1,1);
xes = 1:100;
plot(xes,laserpred,xes,predictpartTuned);
legend({"Correct Values","Predicted Values (tuned NN)"});
xlabel("Time Step");
ylabel("Intensity");
subplot(2,1,2);
plot(xes,laserpred,xes,predictpartUnTuned);
legend({"Correct Values","Predicted Values (Untuned NN)"});
