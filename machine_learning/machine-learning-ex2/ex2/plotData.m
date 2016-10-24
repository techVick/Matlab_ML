function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
pos=find(y==1);
neg=find(y==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,...
    'MarkerSize',7);
% Create New Figure
hold on;
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y',...
    'MarkerSize',7);
xlabel('Exam 1 Score')
ylabel('Exam 2 Score')
title('Figure 1: Scatter plot of training data')

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);








% =========================================================================



hold off;

end
