preprocessing;

%Split the data into training and validation sets to fine tune parameters
Xtrainset = Xtrain(1:490);
Xvaliset = Xtrain(491:end);

%Using basic models we want to first see if the lagging and the number of
%neurons changes much in our model on a validation set

laggings = linspace(10,200,(200-10)/10 + 1); %Here the formula (b-a)/D + 1 gives increments of D
neurons = linspace(10,50,(50-10)/10+1);

%First we check the effect of the lagging
iterations = 1;

%Error Vectors
LagErrors = zeros(1,length(laggings));
rng('default')
for it = 1:iterations,
    lagindex = 1
    for lag = laggings,
        
        [X,Y] = getTimeSeriesTrainData(Xtrainset,lag);
        [Xvali,Yvali] = getTimeSeriesTrainData(Xvaliset,lag);
        
        ptr = con2seq(X);
        ttr = con2seq(Y);
        
        nn = feedforwardnet(10,'trainlm');
        nn.trainParam.epochs = 50;
        nn = train(nn,ptr,ttr);
        
        
        %initializing prediction array based on trainingset data to
        %make first predictions
        %{
        datapredict = [];
        datapredict(1,:) = Xtrainset(end-lag+1:end,:)';
        predictresult = Xtrainset(end-lag+1:end,:)';
        
        for i = 1:length(Xvaliset);
            datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
            ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
            tt = sim(nn, ptest); %Predict the next value
            predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
        end
        
        predictpart = predictresult(:,lag+1:end)';
        %}
        
        Y = sim(nn,con2seq(Xvali));
        LagErrors(lagindex) = LagErrors(lagindex) + mse(nn,Yvali,cell2mat(Y));
        lagindex = lagindex + 1
    end
end

LagErrors = LagErrors / iterations;
Emin = min(LagErrors);
Lagindex = find(LagErrors == Emin);
figure;
plot(laggings,LagErrors);
hold on;
yline(Emin,"--b");
hold off;
xlabel("Lag");
ylabel("Average Error");
xticks(laggings);
title("Effect of Lag on MSE");


%We now check the effect of the number of neurons.  In an earlier
%iteration, we found that the error scales after ~ 10 neurons, so we check
%in a neighborhood around 10 to see if there are any local minima before
%the scale increase
neurons = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300];
lag = laggings(Lagindex);
NeuronErrors = zeros(1,length(neurons));

for it = 1:iterations;
    fprintf("Iteration: %d",it);
    neuronindex = 1
    for neuron = neurons;
        
        [X,Y] = getTimeSeriesTrainData(Xtrainset,lag);
        [Xvali,Yvali] = getTimeSeriesTrainData(Xvaliset,lag);
        
        ptr = con2seq(X);
        ttr = con2seq(Y);
        
        nn = feedforwardnet(neuron,'trainlm');
        nn.trainParam.epochs = 50;
        nn = train(nn,ptr,ttr);
        
        %{
        %initializing prediction array based on trainingset data to
        %make first predictions
        datapredict = [];
        datapredict(1,:) = Xtrainset(end-lag+1:end,:)';
        predictresult = Xtrainset(end-lag+1:end,:)';
        
        for i = 1:length(Xvaliset);
            datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
            ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
            tt = sim(nn, ptest); %Predict the next value
            predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
        end
        
        predictpart = predictresult(:,lag+1:end)';
        %}
        Y = sim(nn,con2seq(Xvali));
        NeuronErrors(neuronindex) = NeuronErrors(neuronindex) + mse(nn,Yvali,cell2mat(Y));
        neuronindex = neuronindex + 1
    end
end

NeuronErrors = NeuronErrors / iterations;
NEmin = min(NeuronErrors);
NeuronIndex = find(NeuronErrors == NEmin);
figure;
plot(neurons,NeuronErrors);
hold on;
yline(NEmin,"--b");
hold off;
xlabel("Number of Neurons");
ylabel("Average Error");
xticks(neurons);
title("Error trend as Number of Neurons Increases");

%We found that for 40 neurons the MSE was minimized so now we check for the
%addition of another layer

layerlist = [5:15,20,40];
LayerErrors = zeros(1,length(layerlist));
iterations = 10;
for it = 1:iterations;
    fprintf("Iteration: %d",it);
    layerindex = 1
    for layer2 = layerlist;
        
        
        [X,Y] = getTimeSeriesTrainData(Xtrainset,lag);
        [Xvali,Yvali] = getTimeSeriesTrainData(Xvaliset,lag);
        
        ptr = con2seq(X);
        ttr = con2seq(Y);
        
        
        nn = feedforwardnet([11,layer2],'trainlm');
        nn.trainParam.epochs = 50;
        nn = train(nn,ptr,ttr);
        
        %{
        %initializing prediction array based on trainingset data to
        %make first predictions
        datapredict = [];
        datapredict(1,:) = Xtrainset(end-lag+1:end,:)';
        predictresult = Xtrainset(end-lag+1:end,:)';
        
        for i = 1:length(Xvaliset);
            datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
            ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
            tt = sim(nn, ptest); %Predict the next value
            predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
        end
        
        
        predictpart = predictresult(:,lag+1:end)';
        %}
        Y = sim(nn,con2seq(Xvali));
        LayerErrors(layerindex) = LayerErrors(layerindex) + mse(nn,Yvali,cell2mat(Y));
        layerindex = layerindex + 1
    end
end

LayerErrors = LayerErrors / iterations;
LEmin = min(LayerErrors);
LayerIndex = find(LayerErrors == LEmin);
plot(layerlist,LayerErrors);
hold on;
yline(LEmin,"--b");
hold off;
xlabel("Number of Neurons in Second Layer");
ylabel("Average Error");
xticks(layerlist);
title("Including a Second Layer Gives Little to No Improvement");


fprintf("With one layer of 11 neurons the MSE is: %d\nWith two layers, one with 11 neurons and the other with 8 the MSE is: %d\n",NEmin,LEmin);

%We find that the error actually increased with the best performing number
%of neurons in the second layer, so we abandon it.

%To fine tune further, we might want to check interactions with laggings
%and neurons, so we perform another optimization near the two local minima.

neuronlist = 7:10;
laglist = [60,70,80];
Errors = zeros(length(neuronlist),length(laglist));

for it = 1:iterations
    fprintf("Iteration: %d\n",it);
    NIndex = 1;
    for neuron = neuronlist
        LIndex = 1;
        for lag = laglist
            [X,Y] = getTimeSeriesTrainData(Xtrainset,lag);
            
            ptr = con2seq(X);
            ttr = con2seq(Y);
            
            
            nn = feedforwardnet([11,neuron],'trainlm');
            nn.trainParam.epochs = 50;
            nn = train(nn,ptr,ttr);
            
            
            %initializing prediction array based on trainingset data to
            %make first predictions
            datapredict = [];
            datapredict(1,:) = Xtrainset(end-lag+1:end,:)';
            predictresult = Xtrainset(end-lag+1:end,:)';
            
            for i = 1:length(Xvaliset);
                datapredict(i,:) = predictresult(i:end); %Start with data used to predict the next value
                ptest = con2seq(datapredict(i,:)'); %convert it to a useful form
                tt = sim(nn, ptest); %Predict the next value
                predictresult = [predictresult, cell2mat(tt)]; %add the predicted value to the vector
            end
            
            predictpart = predictresult(:,lag+1:end)';
            Errors(NIndex,LIndex) = Errors(NIndex,LIndex) + mse(nn,Xvaliset,predictpart);
            LIndex = LIndex + 1;
        end
        NIndex = NIndex + 1;
    end
end

Errors = Errors / iterations;
h = heatmap(Errors);
h.Title = "MSE for combinations of depth and lagging";
h.XLabel = "Lag";
h.YLabel = "Number of Neurons in Second Layer";
h.XData = laglist;
h.YData = neuronlist;

min = max(Errors(1,:)); %initialize min as max of one row of Error matrix
loc = [0 0];
for i = 1:length(Errors(:,1)),
    for j = 1:length(Errors(1,:)),
        if Errors(i,j) < min,
            min = Errors(i,j);
            loc = [i j];
        end
    end
end

%Having found a minima at a lag of 70 with 12 neurons, we now try to
%predict the future

