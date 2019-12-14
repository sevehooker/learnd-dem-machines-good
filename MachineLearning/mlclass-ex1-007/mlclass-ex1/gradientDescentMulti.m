function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    hypothesis = X*theta;
    diff = hypothesis - y;

    for(i = 1:size(X,2)),
         total = sum(diff' * X(:,i));
         theta(i) = theta(i) - alpha / m * total;
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    if(iter > 1 && J_history(iter) > J_history(iter-1))
        printf('cost increase: %f, %f\n', J_history(iter-1), J_history(iter));
        break;
    endif
end

end
