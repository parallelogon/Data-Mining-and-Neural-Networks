load("breast.mat");

%First make a NN with  and train using SCG
net = feedforwardnet(30,'trainscg');
net = train(net,trainset.',labels_train.');
%Then use evidence to optimize
[net,a,b] = evidence(net,trainset,labels_train,500);
