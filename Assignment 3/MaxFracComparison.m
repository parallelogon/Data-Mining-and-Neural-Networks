load choles_all;
%{
X = randn(size(p));
Pstd = mapstd(p);
Xstd = mapstd(X);

coefs = pca(Xstd);
%}



Xstd = mapstd(p);
i = 1;
fracs = linspace(0,1,(1-0)/0.1 + 1);
for maxfrac = fracs
    [Z,PS] = processpca(Xstd,maxfrac);
    numberkept(i) = PS.yrows;
    Xhat = processpca('reverse',Z,PS);
    RMSE_MF(i) = (sqrt(mean(mean((Xstd-Xhat).^2))));
    i = i + 1;
end

plot(fracs,RMSE_MF);
xlabel("Cutoff Fraction");
ylabel("Standardized RMSE");
title("Increase in RMSE as cutoff value is raised");