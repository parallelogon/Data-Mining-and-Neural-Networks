load curses.mat
ss = 1;
sigma = 1;
subplot(2,2,1);
for n = 1:7;
    plot(1:7,time_poly(:,n,ss,sigma));
    hold on;
end
legend({"P = 1","P = 2","P = 3",'P = 4','P = 5','P = 6','P = 7'},'location','northwest');
xlabel("Dimension");
ylabel("Training Time");
title("Training Time Increases for Polynomial Fit");
hold off

subplot(2,2,2);
for n = 1:7;
    plot(1:7,time_nn(:,n,ss,sigma));
    hold on;
end
xlabel("Dimension");
ylabel("Training Time");
title("Training Time Trend for Neural Network Fit");
hold off

subplot(2,2,3);
for n = 1:7;
    plot(1:7,rmse_poly_test(:,n,ss,sigma));
    hold on;
end
xlabel("Dimension");
ylabel("Test RMSE");
title("Error Trend for Polynomial Fit");
hold off

subplot(2,2,4);
for n = 1:7;
    plot(1:7,rmse_nn_test(:,n,ss,sigma));
    hold on;
end
xlabel("Dimension");
ylabel("Test RMSE");
title("Error Trend for NN Fit");
hold off