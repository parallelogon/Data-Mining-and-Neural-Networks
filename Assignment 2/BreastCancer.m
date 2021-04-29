load('breast.mat')


%First we visualize the data with a boxplot noting columns 4 and 24
subplot(1,2,1);
boxplot(trainset(labels_train == 1,:));
xlabel('Column')
ylabel('Values')
title('Values corresponding to +1');
xtickangle(270);

subplot(1,2,2);
boxplot(trainset(labels_train == -1,:));
xlabel('Column')
ylabel('Values')
title('Values corresponding to -1');
xtickangle(270);

%Lets make a linear classifier along these two axis
Xlinear = [trainset(:,4) trainset(:,24)].';
Ylinear = labels_train.';
Ylinear(:,Ylinear == -1) = 0;

Xlineartest = [testset(:,4) testset(:,24)].';

linearnet = perceptron;
linearnet = train(linearnet,Xlinear,Ylinear);

theta = linearnet.iw{1,1};
theta_0 = linearnet.b{1};
Y = sim(linearnet,Xlineartest);

err = mse(linearnet,labels_test,Y);


%The data is almost linearly seperable in these two directions
scatter(trainset(labels_train == -1,4),trainset(labels_train == -1,24),'x');
hold on;
scatter(trainset(labels_train == 1,4),trainset(labels_train == 1,24),'o');
hold on;
plotpc(theta,theta_0);
hold off;
xlabel("Column 4");
ylabel("Column 24");
title("Data is nearly linearly seperable with MSE of 1.73");

%Now we see if we can improve using a subset of the data as training, a
%subset as validation and a neural network

%since this is a clasification problem, we randomly choose 20% of the data
valipicks = rand(1,400) >= 0.8;
trainpicks = ones(1,400) - valipicks;

Xtrain = trainset(trainpicks==1,:);
Ytrain = labels_train(trainpicks==1,:);

Xvali = trainset(valipicks==1,:);
Yvali = labels_train(valipicks==1,:);



%adjust the number of neurons/depth of NN using validation set
%note that 80 neurons is massively overparameterized

iterations = 10;
neuronlist = [1:10,20,40];
algorithms = ["trainbr","traingd","trainlm","trainscg"];
Errors = zeros(length(algorithms),length(neuronlist));

for it = 1:iterations
    algIndex = 1
    fprintf("Iteration: %d\n",it);
    for algorithm = algorithms
        
        nIndex = 1;
        for neuron = neuronlist
            
            nn = feedforwardnet(neuron,algorithm);
            nn = train(nn,Xtrain.',Ytrain.');
            Y = sim(nn,Xvali.');
            Incorrect = Y.*Yvali.' < 0;
            %Errors(algIndex,nIndex) = Errors(nIndex) + mse(nn,Yvali,Y);
            Errors(algIndex,nIndex) = Errors(algIndex,nIndex) + sum(Incorrect);
            nIndex = nIndex + 1;
        end
        algIndex  = algIndex + 1;
    end
end

Errors = Errors / iterations;

%find row column of minimum error measurement
Emin = max(Errors(1,:)); %initialize min as max of one row of Error matrix
loc = [0 0];
for i = 1:length(Errors(:,1)),
    for j = 1:length(Errors(1,:)),
        if Errors(i,j) < Emin,
            Emin = Errors(i,j);
            loc = [i j];
        end
    end
end

h = heatmap(Errors);
h.Title = "Average Number Incorrect for Different Neuron and Algorithm combinations";
h.XLabel = "Number of Neurons";
h.YLabel = "Algorithm";
h.XData = neuronlist;
h.YData = algorithms;
%{
plot(neuronlist,Errors);
hold on;
yline(Emin,'--b');
hold off;
xlabel("Number of Neurons");
ylabel("MSE");
xticks(neuronlist);
title("Selecting Lowest MSE for Number of Neurons in the First Layer");
%}

%We find the lowest error with 2 perceptron in the hidden layer using
%trainbr
%We can then check for deeper layers


Errors2 = zeros(1,length(neuronlist));
alg = algorithms(loc(1));
firstLayer = neuronlist(loc(2));

for it = 1:iterations
    fprintf("Iteration %d\n",it);
    nIndex = 1;
    for neuron = neuronlist
        nn = feedforwardnet([firstLayer,neuron],alg);
        nn = train(nn,Xtrain.',Ytrain.');
        Y = sim(nn,Xvali.');
        Incorrect = Y.*Yvali.' < 0;
        Errors2(nIndex) = Errors2(nIndex) + sum(Incorrect);
        nIndex = nIndex + 1;
    end
end

Errors2 = Errors2 / iterations;
Emin2 = min(Errors2);

plot(neuronlist,Errors2);
hold on;
yline(Emin2,'--b');
hold off;
xlabel("Number of Neurons");
ylabel("MSE");
xticks(neuronlist);
title("Selecting Lowest MSE for Number of Neurons in the Second Layer");

%We train a NN with one hidden layer for the total dataset now

nn = feedforwardnet(2,'trainbr');
nn = train(nn,trainset.',labels_train.')
Y = sim(nn,testset.');
MSE = mse(nn,labels_test,Y);
Incorrect = Y.*labels_test.' < 0;
numWrongNN = sum(Incorrect)


linearnet = perceptron;
linearnet = train(linearnet,Xlinear,Ylinear);

theta = linearnet.iw{1,1};
theta_0 = linearnet.b{1};
Yy = sim(linearnet,Xlineartest);
MSELinear = mse(linearnet,labels_test,Yy);
LIncorrect = Yy.*labels_test.' < 0;
numWrong = sum(LIncorrect);


scatter(testset(labels_test == -1,4),testset(labels_test == -1,24),'+','blue');
hold on;
scatter(testset(labels_test == 1,4),testset(labels_test == 1,24),'o','green');
hold on;
scatter(testset(Incorrect,4),testset(Incorrect,24),200,'X','red');
hold on;
plotpc(theta,theta_0);
hold off;
xlabel("Column 4");
ylabel("Column 24");
titletext = sprintf("Data is nearly linearly seperable");
subtitletext = sprintf("%d incorrect for single perceptron, %d incorrect for NN",numWrong,numWrongNN);
title({titletext;subtitletext});