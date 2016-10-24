function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

g1 = 1.0 ./ (1.0 + exp(-z));
a=size(g1,1);
b=size(g1,2);
for i=1:a
    for j=1:b
g(i,j)=g1(i,j)*(1-g1(i,j));
    end
end












% =============================================================




end
