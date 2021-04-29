%Returns the transformation matrix with i largest compenents kept
function [Xhat,rmse] = pcamaker(X,i)
X = X;
mu = mean(X);
X = X - mu;


Sigma = cov(X.');
[E,d] = eig(Sigma);
eigenvals = diag(d);
E = E(:,flip(1:length(eigenvals)));
Ep = E(:,1:i).';
Fp = Ep.';
Z = Ep*X;
Xhat = Fp*Z + mu;
rmse = (sqrt(mean(mean((X-Xhat).^2))));
end