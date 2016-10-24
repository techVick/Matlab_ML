function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];        
a1=X;
z2=a1*transpose(Theta1);
a2=sigmoid(z2);
m1 = size(a2, 1);
a2 = [ones(m1, 1) a2];
z3=a2*transpose(Theta2);
a3=sigmoid(z3);
hx=a3;
y1=zeros(5000,10);
for i=1:m
    y2=y(i,1);
    y1(i,y2)=1;
end
for i=1:m
    for j=1:num_labels
    J=J+(1/m)*(-y1(i,j)*log(hx(i,j))-(1-y1(i,j))*log(1-hx(i,j)));
    end
end
a=size(Theta1,1);
b=size(Theta1,2);
c=size(Theta2,1);
d=size(Theta2,2);

for i=1:a
    Theta1(i,1)=0;
end
for i=1:c
    Theta2(i,1)=0;
end
reg1=0;
for i=1:a
    for j=1:b
        reg1=reg1+(Theta1(i,j)*Theta1(i,j));
    end
end
reg2=0;
for i=1:c
    for j=1:d
        reg2=reg2+(Theta2(i,j)*Theta2(i,j));
    end
end

reg=(lambda/(2*m))*(reg1+reg2);
J=J+reg;
% -------------------------------------------------------------
load('ex4data1.mat');
load('ex4weights.mat');
DEL1=zeros(25,400);
DEL2=zeros(10,25);
a1=X;
Theta1=Theta1(:,2:401);
Theta2=Theta2(:,2:26);
z2=a1*transpose(Theta1);
a2=sigmoid(z2);
z3=a2*transpose(Theta2);
a3=sigmoid(z3);
for i=1:m
delta3=transpose(a3(i,:))-transpose(y1(i,:));
delta_b=(transpose(Theta2)*(delta3));
delta2=delta_b.*(transpose(sigmoidGradient(z2(i,:))));
DEL1=DEL1+delta2*a1(i,:);
DEL2=DEL2+delta3*a2(i,:);
end
Theta1_grad=(DEL1/m)+(lambda/m)*Theta1;
Theta2_grad=(DEL2/m)+(lambda/m)*Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
