function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CC=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
dd=size(CC);
dd=dd(1,1);
err=0;
M=zeros(64,3);
for i=1:dd
    for j=1:dd
    C=CC(i,1);
    sigma=CC(j,1);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
err1=mean(double(predictions ~= yval));
k=(8*i)+j-8;
M(k,1)=err1;
M(k,2)=C;
M(k,3)=sigma;
if err1<err
    Cfinal=C;
    sigmaF=sigma;
end
err=err1;
    end
end
data=M(:,1);
[minNum, minIndex] = min(data(:));
[row, col] = ind2sub(size(data), minIndex);
C=M(row,2);
sigma=M(row,3);
% =========================================================================

end
