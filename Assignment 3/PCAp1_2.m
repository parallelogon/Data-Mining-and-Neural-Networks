load choles_all;
randMat = randn(21,264);

randMat = mapstd(randMat);
p = mapstd(p);

RMSE_rand = pcarmse(randMat);
RMSE_P = pcarmse(p);

subplot(1,2,1);
plot(1:length(RMSE_P),RMSE_P);
xlabel("Number of PC's");
ylabel("RMSE");
title("Sharp Decrase in RMSE for Correlated Data");
subplot(1,2,2);
plot(1:length(RMSE_rand),RMSE_rand);
xlabel("Number of PC's");
ylabel("RMSE");
title("Random Data Gradually Decreases in RMSE");