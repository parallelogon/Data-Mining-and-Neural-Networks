%% perform unsupervised learning with SOM  

% Marco Signoretto, March 2011
% Modified Zach 2019

close all;
clear all;
clc;

% first we import data
load banana.mat;

%Arrange the data in rows for easier input
X = X.';

% we then initialize the SOM with hextop as topology function
% ,linkdist as distance function and gridsize 5x5

%{
     The topology function TFCN can be HEXTOP, GRIDTOP, or RANDTOP.
     The distance function can be LINKDIST, DIST, or MANDIST.
 %}
net = newsom(X,[5 5],'hextop','dist'); 

% plot the data distribution with the prototypes of the untrained network
%a vector of ones for our plot

Z = ones(1,length(X(2,:)));
figure;
plot(X(1,:),X(2,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

% finally we train the network and see how their position changes
net.trainParam.epochs = 100;
net = train(net,X);
figure;
plot(X(1,:),X(2,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

figure;
plotsomhits(net,X);