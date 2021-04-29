X = randn(50,500);

%X = randn(3,5);
X = X.';
coefs = pca(X);

mu = mean(X);
X = X - mu;

Sigma = cov(X);
[E,d] = eig(Sigma);
eigenvals = diag(d);
E = E(:,flip(1:length(eigenvals)));

for i = 1:length(eigenvals)
    Ep = E(:,1:i);
    Fp = Ep.';
    Xhat = X*Ep*Fp;
    RMSE(i) = (sqrt(mean(mean((X-Xhat).^2))));
end


plot(1:length(eigenvals),RMSE);
xlabel("Number of PC's");
xticks(1:length(eigenvals));
ylabel("RMSE");
title("Decrease in Error for Large Number of PC's");