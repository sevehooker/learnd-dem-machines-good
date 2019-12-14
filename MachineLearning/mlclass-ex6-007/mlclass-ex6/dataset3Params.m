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
steps = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%steps = [0.01; 1; 3];
m = length(steps);
x1 = [1 2 1]; x2 = [0 4 -1];

err = 10000;

for i = 1:m
	for j = 1:m
		model = svmTrain(X, y, steps(i), @(x1, x2) gaussianKernel(x1, x2, steps(j)));
		predictions = svmPredict(model, Xval);
		cur_err = mean(double(predictions ~= yval));

		if err > cur_err
			err = cur_err;
			C = steps(i);
			sigma = steps(j);
		end
	end
end
% =========================================================================

end
