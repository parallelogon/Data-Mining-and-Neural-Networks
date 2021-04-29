load threes -ascii;

%{
chooses and displays a random datum
i = randi(length(threes))
colormap('gray');
imagesc(reshape(threes(i,:),16,16),[0,1]);
%}

%Finding the mean of digit 3
Mean3 = mean(threes(3,:));

Sigma = cov(threes);

[E,d] = eig(Sigma);
D = diag(d);
D = D(flip(1:length(D)));
plot(D);
xlabel("Eigenvalue Number");
ylabel("Eigenvalue");
title("Eigenvalues from lowest to highest")

%Compressing the dataset
compress1 = pcamaker(threes,1);
compress2 = pcamaker(threes,2);
compress3 = pcamaker(threes,3);
compress4 = pcamaker(threes,4);


colormap('gray');
subplot(2,2,1);
imagesc(reshape(compress1(3,:),16,16),[0,1]);
title('One PC');
subplot(2,2,2);
imagesc(reshape(compress2(3,:),16,16),[0,1]);
title("Two PC's");
subplot(2,2,3);
imagesc(reshape(compress3(3,:),16,16),[0,1]);
title("Three PC's");
subplot(2,2,4);
imagesc(reshape(compress4(3,:),16,16),[0,1]);
title("Four PC'S");
sgtitle("Rectonstructions of The Number Three");

%This for loop computes reconstruction error from PCA's
rmse = 0;
for k = 1:50
    Xhat = pcamaker(threes,k);
    rmse(k) = (sqrt(mean(mean((threes-Xhat).^2))));
end

plot(1:50,rmse);
xlabel("k");
ylabel("RMSE");
title("Error Decreases for Large Number of PC's");

%for question 5
RMSE256 = (sqrt(mean(mean((threes-pcamaker(threes,256)).^2))));
fprintf("For 256 PC's the RMSE is %d\n",RMSE256);

%VarExplained
varExplained = cumsum(D) - D;

plot(1:50,rmse,1:50,varExplained(1:50)/sum(D));
xlabel("K PC's");
legend({"RMSE","Normalized Sum of Eigenvalues"});