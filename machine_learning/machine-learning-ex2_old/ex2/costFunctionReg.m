function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
z5=size(X);
z4=[];
z6=0;
for j=1:z5(1,2)
   z4=[z4 X(i,j)];
   z6=z6+(theta(j,1)*theta(j,1));
end
z4=transpose(z4);
z3=transpose(theta)*(z4);
J=J+(1/m)*((-y(i,1)*log(sigmoid(z3)))-((1-y(i,1))*log(1-sigmoid(z3))));
grad=grad+((1/m)*(sigmoid(z3)-y(i,1)))*(z4);
z6=z6-(theta(1,1)*theta(1,1));
end
theta(1,1)=0;
J=J+(lambda/(2*m))*z6;
grad=grad+(lambda/m)*theta;
% =============================================================
end
