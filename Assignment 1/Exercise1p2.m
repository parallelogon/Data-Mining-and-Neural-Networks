%Depth of MLP
d = 10;

%Create a noiseless dataset
x = linspace(0,3*pi,1000);
y = sin(x.^2);
Lx = length(x);
%A vector of possible training algorithms with length L and a matrix of
%Scores to hold the intercept, slope, and r value of the fitted classifier
algs = ["traingd","traingda","traincgf","traincgp","trainbfg","trainlm"]
L = length(algs);
Scores = zeros(L,3);
Y = zeros(L,Lx);
%Loops through and trains a network using algorithim i, stores score values
%in the matrix Scores
for i = 1:L
    nettmp = feedforwardnet(d,algs(i));
    nettmp = train(nettmp,x,y);
    Y(i,:) = sim(nettmp,x);
    [Scores(i,1),Scores(i,2),Scores(i,3)] = postreg(Y(i,:),y);
end



%Recreate y with random noise
yrnd = y + randn([1,Lx])

%make a new scores matrix
ScoresRnd = zeros(L,3);
Y2 = zeros(L,Lx);
%Loops through and trains a network using algorithim i, stores score values
%in the matrix with new Scores
for i = 1:L
    nettmp = feedforwardnet(d,algs(i));
    nettmp = train(nettmp,x,yrnd);
    Y2(i,:) = sim(nettmp,x);
    [ScoresRnd(i,1),ScoresRnd(i,2),ScoresRnd(i,3)] = postreg(Y2(i,:),yrnd);
end


%Plots the r value by training algorithm and randomized vs unrandomized
%data
figure;
plot(1:6,Scores(:,3),1:6,ScoresRnd(:,3));
legend({'unrandomized data','randomized data'},'Location','Southeast')
ylabel("r value");
xticklabels(algs);
xtickangle(45);
title("function approximation and data fit for y = sin(x^2)");

%Plots the noisy data and real function by algorithm type for randomized
%data
figure;
for i=1:6;
    subplot(6,2,2*i-1);
    plot(x,y,x,Y(i,:));
    subplot(6,2,2*i);
    plot(x,yrnd,x,Y2(i,:));
end

%Now test the affect on the trainlm function with increasing numbers of
%datapoints and increasing noise
d = 3
X = cell(1,d);
Y = cell(d,d);

Ynet = cell(d,d);


d1 = 0;
d2 = 0;

scoresmat = cell(d,d);
for i = 1:d;
    for j = 1:d;
        X{i} = linspace(0,3*pi,75*(2^i));
        Y{j,i} = sin(X{i}.^2) + randn([1,length(X{i})])/(d-j+1);
        nettmp = feedforwardnet(10,"trainlm");
        nettmp = train(nettmp,X{i},Y{j,i});
        Ynet{j,i} = sim(nettmp,X{i});
        [d1,d2,scoresmat{j,i}] = postreg(Ynet{j,i},Y{j,i});
    end
end

for i = 1:d;
    for j = 1:d;
        subplot(d,d,i+d*j-d);
        plot(X{i},Y{j,i},X{i},Ynet{j,i});
        title(sprintf('r = %.2f, s = %.2f',scoresmat{j,i},(d-j+1)*length(X{i})));
    end
end

load("data_personal_regression_problem.mat")
Tnew = (7*T1 + 7*T2 + 4*T3 + 3*T3 + 3*T3)/(7+7+4+3+3);

%First we randomize the dataset and pick out training validation and test
%sets

indices = randperm(length(X1))
X1X2 = [X1 X2]
XTraining = X1X2(indices(1:1000),:).'
YTraining = Tnew(indices(1:1000)).'
XVal =  X1X2(indices(1001:2000),:).'
YVal = Tnew(indices(1001:2000)).'
XTest =  X1X2(indices(2001:3000),:).'
YTest = Tnew(indices(2001:3000)).'

%Because of universal approximation theorem, we only need to find new
%number of neurons in one hidden layer, we will use validation set to
%approximate it

%depth = 2.^(1:10)
depth = 1:40 %used for finer searching

Myscores = zeros(length(depth),1)

for i = depth
    %Creating and training the NN
    net = feedforwardnet(i,"trainlm");
    net = train(net,XTraining,YTraining);
    
    %Validating
    Ytmp = sim(net,XVal);
    Myscores(i) = mse(net,YVal,Ytmp);
end

[minMSE,id] = min(Myscores);
netfinal = feedforwardnet(id,"trainlm");
netfinal = train(netfinal,XTraining,YTraining);
Yfinal = sim(netfinal,XTest);
Result = mse(netfinal,Yfinal,YTest);
fprintf("Depth: %d\nMSE: %d\n", id,Result);


figure;
subplot(1,2,1);
plot3(XTest(1,:),XTest(2,:),Yfinal,'.b');
xlabel("X1");
ylabel("X2");
zlabel("f(X1,X2)");
title("Simulation Results");
subplot(1,2,2);
plot3(XTest(1,:),XTest(2,:),YTest,'.g');
xlabel("X1");
ylabel("X2");
zlabel("Tnew");
title("True Values");
figure;
subplot(2,1,1);
plot(depth,Myscores);
xlabel("Depth");
ylabel("r value");
subplot(2,1,2);
plot(1:1000,YTest, 1:1000, Yfinal);
legend({"True", "Output"});
xlabel("Datapoint Number");
ylabel("Output")