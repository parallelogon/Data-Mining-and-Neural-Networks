function rmse = pcarmse(X)
X = X.';
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
    rmse(i) = (sqrt(mean(mean((X-Xhat).^2))));
end
end