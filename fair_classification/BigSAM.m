function [f_vec,g_vec,time_vec,x,acc_vec,p_vec] = BigSAM(fun_f,grad_f,grad_g,fun_g,acc_p_rule,param,x0)
% BiG-SAM in "A first order method for solving convex bilevel optimization problems",
% S. Sabach and S. Shtern, SIOPT 2017
% The update rule is given by: 
%
% y_{k+1} = \Pi_{Z}(x_k-\eta_g \nabla g(x_k)),
% z_{k+1} = x_k - \eta_f\nabla f(x_k),
% x_{k+1} = \alpha_{k+1} z_{k+1} + (1-\alpha_{k+1}) y_{k+1}
%

% param definition
eta_f = param.eta_f;
eta_g= param.eta_g;
lambda = param.lam;
gamma = param.gamma;

maxiter = param.maxiter;
maxtime = param.maxtime;

% initialization
x = x0;
iter = 0;
f_vec = fun_f(x);
g_vec = fun_g(x);

[acc_vec, p_vec] = acc_p_rule(x);
time_vec = 0;

tic;
while iter <= maxiter
    iter = iter+1;
    x_lo = x-eta_g*grad_g(x);
    % Projection to simplex
    x_lo = ProjectOntoL1Ball(x_lo,lambda);

    x_up = x-eta_f*grad_f(x);
    
    % Averaging
    alpha = min([2*gamma/iter,1]);
    x = alpha*x_up + (1-alpha)*x_lo;
    cpu_t1 = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(x)];
    
    [acc, p] = acc_p_rule(x);
    p_vec = [p_vec;p];
    acc_vec = [acc_vec;acc];
    time_vec = [time_vec;cpu_t1];
    
    if mod(iter,5000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t1>maxtime
        break
    end
end

end