numbers_train = [5000,10000,15000] ;                            % training set size
n_test  = 500  ;                            % test set size
Ln = length(numbers_train);

samples_train = [0.1,0.5,1] ;                              % noise standard deviation of the training set
s_test  = .0 ;   
Ls = length(samples_train);

Dmax = 7;
dimension = 1:Dmax;
R = 1;

Pmax = 7;
polydegrees = 1:Pmax;
n_neurons = [4 2];

%initialize data in a vector with dimensions
%[dimension[polydegrees[ntrain[s_train]]]]
rmse_poly_test = zeros(Dmax,Pmax,Ln,Ls);
rmse_nn_test = zeros(Dmax,Pmax,Ln,Ls);
time_poly = zeros(Dmax,Pmax,Ln,Ls);
time_nn = zeros(Dmax,Pmax,Ln,Ls);

for d = dimension;
    for p = polydegrees;
        for n_train = numbers_train;
            for s_train = samples_train
                i = find(numbers_train == n_train);
                j = find(samples_train == s_train);
                
Train_input  = randsphere(n_train, d, R) ;                      % samples training set on the hyper-sphere 
Test_input   = randsphere(n_test,  d, R) ;                      % samples test set on the hyper-sphere

Train_norms  = sqrt(sum(Train_input.^2,2)) ;                    % computes euclidean norm of each datapoint of the training set
Test_norms   = sqrt(sum(Test_input.^2, 2)) ;                    % computes euclidean norm of each datapoint of the test set

Train_output = sinc(Train_norms) ;                              % computes the cardinal sinus of each norm of the training set
Test_output  = sinc(Test_norms) ;                               % computes the cardinal sinus of each norm of the test set

Train_output_noisy = Train_output + s_train*randn(n_train,1) ;  % adds eventual noise to the training set   (won't change anything if s=0)
Test_output_noisy  = Test_output  + s_test *randn(n_test, 1) ;  % adds eventual noise to the test set       (won't change anything if s=0)


%Put this in a for loop
tic ;                                                                           % starts the times for the polynomial
mdl_poly = polyfitn(Train_input,Train_output_noisy,p) ;                         % (training) performs the multi-dimensional polynomial regression on the training set
time_poly(d,p,i,j) = toc ;                                                               % stops the timer and saves the time of training the polynomial

Poly_train_output = polyvaln(mdl_poly,Train_input) ;                            % evaluates the test inputs on the trained polynomial model
rmse_poly_train = 1/n_train*sqrt(sum((Poly_train_output-Train_output).^2)) ;    % computes root mean square error on the test set
Poly_test_output = polyvaln(mdl_poly,Test_input) ;                              % evaluates the test inputs on the trained polynomial model
rmse_poly_test(d,p,i,j) = 1/n_test*sqrt(sum((Poly_test_output-Test_output).^2)) ;        % computes root mean square error on the test set

%This as well
mdl_nn = feedforwardnet(n_neurons,'trainlm') ;                              % creates the feedforward neural network
mdl_nn.trainParam.showWindow = false ;                                      % avoid plotting output window
tic ;                                                                       % starts the timer for the training of the neural network
mdl_nn = train(mdl_nn,Train_input',Train_output') ;                         % trains the network
time_nn(d,p,i,j) = toc ;                                                             % stops the timer and saves the training time of the neural network

NN_train_output = mdl_nn(Train_input') ;                                    % evaluates the network on the test set
rmse_nn_train = 1/n_test*sqrt(sum((NN_train_output'-Train_output).^2)) ;    % computes root mean square error on the test set
NN_test_output = mdl_nn(Test_input') ;                                      % evaluates the network on the test set
rmse_nn_test(d,p,i,j) = 1/n_test*sqrt(sum((NN_test_output'-Test_output).^2)) ;       % computes root mean square error on the test set

vol           = pi^(d/2)*R^d/gamma(d/2+1) ;                                         % volume of the domain hyper-sphere
n_params_poly = nchoosek(d+p,p) ;                                                   % number of parameters for the polynomial model
n_params_nn   = sum(n_neurons) + sum([d n_neurons 1 0].*[0 d n_neurons 1]) ;        % number of parameters for the neural network model
            end
        end
    end
end