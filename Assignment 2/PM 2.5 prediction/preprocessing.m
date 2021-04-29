%% preprocessing by substituting the missing data with their corresponding interpolation values
load('shanghai2017.mat');

X = shanghai2017;
indices = find(X == -999);

% each missing data is substituted by the order-one interpolation on its
% nearest six observed data.
for i = [1:6]
    copy_indices = indices;
    copy_indices(i) = [];
    all_indices = [1:length(X)];
    all_indices(copy_indices) = [];
    index = find(all_indices == indices(i));
    x = [];
    y = [];
    for j = [1:3,5:7]
        x = [x, all_indices(index+(j-4))];
        y = [y, X(all_indices(index+(j-4)))];
    end
    p = polyfit(x, y, 1);
    f = polyval(p, indices(i));
    X(indices(i)) = f;
end

% the first 700 are training data and the rest 300 are test data
Xtrain = X(1:700);
Xpred = X(701:1000);

% plot the preprocessed data set
figure
plot(Xtrain)
hold on
idx = 700:1000;
plot(idx,[Xtrain(end);Xpred],'-')
hold off
xlabel("Data (in Hour)")
ylabel("PM 2.5 Concentration Index")
title("PM 2.5 Value of Shanghai, China in 2017")
legend(["Training data" "Test Data"])