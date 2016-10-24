function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hx=transpose(theta)*transpose(X);
hx=transpose(hx);
%hx=1.0 ./ (1.0 + exp(-z));
J=((1/(2*m))*transpose(hx-y)*(hx-y));
theta(1,1)=0;
J=J+(lambda/(2*m))*transpose(theta)*theta;
grad=(1/m)*transpose(X)*(hx-y);
grad=grad+(lambda/(m))*(theta);
% =========================================================================

grad = grad(:);

end
