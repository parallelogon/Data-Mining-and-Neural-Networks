load rings.mat

X = X
scatter(X(Y == 0,1),X(Y == 0,2),'r');
hold on;
scatter(X(Y == 1,1),X(Y == 1,2),'g');
hold on;
scatter(X(Y==2,1),X(Y==2,2),'b');
hold off
legend({"Y = 0","Y = 1","Y = 2"});
title("Visualization of Rings Data");

%separate the data out into groups
X0 = X(Y==0,:);
X1 = X(Y==1,:);
X2 = X(Y==2,:);

%find minima within group
within_group_minima = zeros(1,3);
within_group_minima(1) = min(pdist(X0));
within_group_minima(2) = min(pdist(X1));
within_group_minima(3) = min(pdist(X2));


%Here we find betweeng roup minima
between_group_minima = zeros(1,3);

%Modifying X1 to be the same size as X0 and finding the minimimum pointwise distance between X1 and X0
A = min(size(X0),size(X1));
B = max(size(X0),size(X1));
X1p = [X1.' nan(B(1) - A(1),2).'].'

between_group_minima(1) = min(pdist([X1p X0]));



%modify X2 to be the same size as X0 and finding mimimum pairwise distance
%between X2 and X0
A = min(size(X0),size(X2));
B = max(size(X0),size(X2));
X2p = [X2.' nan(B(1) - A(1),2).'].';

between_group_minima(2) = min(pdist([X2p X0]));
%min(X0,X1).' = [min(X0,X1).',nan((B(1)-A(1)),2).']

%repeating for X1 and X2
A = min(size(X1),size(X2));
B = max(size(X1),size(X2));
X1pp = [X1.' nan(B(1) - A(1),2).'].';

between_group_minima(3) = min(pdist([X2 X1pp]));

%we can conclude that the neighborhoods must include the largest minima
%within the groups and be smaller than the smallest minimum distance
%between the groups in order to properly cluster the data

lower = max(within_group_minima);
upper = min(between_group_minima);
fprintf("epsilon must define a neighborhood within %d, and %d\n",lower,upper);

%to find the best epsilon we need to find an "elbow in a graph for the
%three closest distances

for i = 1:length(X)
    %makes a column vector of X(i)'s and finds their distance
    %simultaneously to all other points by subtracting the two vectors and squaring each component indivudally 
    %(1)then summing the components and square rooting then sorting(2)
    
    squares = (ones(size(X)).*X(i,:) - X).^2; %(1)
    D = sort(sqrt(squares(:,1) + squares(:,2))); %(2)
    
    %We then find the closest three, sum them and add it to our vector of
    %distances
    distances_pre_ascending(i) = sum(D(1:4));
end

plot(sort(distances_pre_ascending));
title("Elbow location for optimal epsilon");

%With optimum about 2.4 we can find our clusters.  Maybe
%we notice that the density of points is approximately ##of points/area of
%points, calculating these values as ##/L with L being the farthest distance between two points in the set
%for X0, X1, and X2 we get 20.25, 17.66, and 12.60 respectively, setting
%NN/epsilon to approximately 20 allows us to distinguish between X0 and X1
[idx,noise] = DBSCAN(X,2.4,1000);

%plot the data
scatter(X(idx == 0,1),X(idx == 0,2),'r');
hold on;
scatter(X(idx == 1,1),X(idx == 1,2),'g');
hold on;
scatter(X(idx==2,1),X(idx==2,2),'b');
hold off
legend({"idx = 0","idx = 1","idx = 2"});
title("Visualization of Rings Data");
%[silh,h] = silhouette(X,idx)