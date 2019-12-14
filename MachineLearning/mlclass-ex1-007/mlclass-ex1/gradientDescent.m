function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
x = X(:,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    hypothesis = theta(1) + theta(2)*x;
    
    diff = hypothesis - y;
    sum1 = sum(diff);
    sum2 = sum(diff' * x);

    theta = [theta(1) - alpha / m * sum1; theta(2) - alpha / m * sum2];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    if(iter > 1 && J_history(iter) > J_history(iter-1))
        printf('broked\n');
    endif
end

end
