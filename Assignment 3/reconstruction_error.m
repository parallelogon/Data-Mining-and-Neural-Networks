%This function projects and rebuilds pca's and computes their error
function reconstruction_error = reconstruct_err(X,k)
Xhat = pcamaker(X,k);
reconstruction_error = (sqrt(mean(mean((X-Xhat).^2))));
end