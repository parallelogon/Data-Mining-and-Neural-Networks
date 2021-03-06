%% Partition Data into Two Clusters
%%
% Randomly generate the sample data.

% Copyright 2015 The MathWorks, Inc.

%rng default; % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2)];

figure;
plot(X(:,1),X(:,2),'.');
title 'Randomly Generated Data';
%%
% There appears to be two clusters in the data.
%%
% Partition the data into two clusters, and choose the best arrangement out of
% five intializations. Display the final output.
opts = statset('Display','final');
[idx,C] = kmeans(X,2,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
%%
% By default, the software initializes the replicates separatly using
% _k_-means++.
%%
% Plot the clusters and the cluster centroids.
figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

