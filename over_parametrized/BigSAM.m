function [f_vec1,g_vec1,time_vec1,x,acc_vec] = BigSAM(fun_f,grad_f,grad_g,fun_g,TSA,param,x0)
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

x = x0;


tic;
% algorithm
iter = 0;
f_vec1 = [];
g_vec1 = [];
time_vec1 = [];
acc_vec = [];
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
    f_vec1 = [f_vec1;fun_f(x)];
    g_vec1 = [g_vec1;fun_g(x)];
    time_vec1 = [time_vec1;cpu_t1];
    % test set accuracy
    [acc_vec] = [acc_vec;TSA(x)];
    if mod(iter,5000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t1>maxtime
        break
    end
end