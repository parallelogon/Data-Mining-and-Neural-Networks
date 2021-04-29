coef = pca(trainset);
xpca = [trainset*coef(:,1) trainset*coef(:,2) trainset*coef(:,3)];
xpcatest = [testset*coef(:,1) testset*coef(:,2) testset*coef(:,3)];

scatter(xpca(labels_train == 1,1),xpca(labels_train==1,2),'+','green');
hold on;
scatter(xpca(labels_train == -1,1),xpca(labels_train == -1,2),'x','red');
hold off;

nnpca = feedforwardnet(2,'trainbr');
nnpca = train(nnpca,xpca.',labels_train.');
Y = sim(nnpca,xpcatest.');
incorrect = Y.*labels_test.' < 0;
sum(incorrect)