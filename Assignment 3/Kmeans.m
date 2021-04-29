load rings.mat;

scatter(X(Y == 0,1),X(Y == 0,2),'r');
hold on;
scatter(X(Y == 1,1),X(Y == 1,2),'g');
hold on;
scatter(X(Y==2,1),X(Y==2,2),'b');
hold off
legend({"Y = 0","Y = 1","Y = 2"});
title("Visualization of Rings Data");

[Idx,C] = kmeans(X,3,'Replicates',5);

x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'Start',C);
    % Assigns each node in the grid to the closest centroid
%%
% |kmeans| displays a warning stating that the algorithm did not converge,
% which you should expect since the software only implemented one
% iteration.
%%
% Plot the cluster regions.
figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title("K means Clusters");
%{
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
%}
hold off;

[silh3,h] = silhouette(X,Idx);
xlabel('Silhouette Value')
ylabel('Cluster')
title("Silhouette Plot showing Poor Separation");