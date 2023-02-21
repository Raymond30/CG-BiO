function [f_vec,g_vec,time_vec,x,acc_vec,p_vec] = DBGD(fun_f,grad_f,grad_g,fun_g,acc_p_rule,param,x0)
% DBGD in "Bi-objective trade-off with dynamic barrier gradient descent",
% C. Gong, X. Liu, and Q. Liu, Neurips 2021
% The update rule is given by 
% x_{k+1} = \Pi_{Z}(x_k - \gamma_k(\nabla f (x_k)+\lambda_k \nabla g (x_k))),
% \lambda_k = \max\{ \frac{\phi(x_k)- \nabla f(x_k)'* \nabla g(x_k)}{\|\nabla g(x_k)\|^2},0 \},
% and \phi(x) = \min\{\alpha (g(x)-\hat{g}), \beta \|\nabla g (x)\|^2\}

stepsize = param.stepsize;
alpha = param.alpha;
beta = param.beta;

lambda = param.lam;

maxiter = param.maxiter;
maxtime = param.maxtime;

x = x0;

tic;
% algorithm
iter = 0;
f_vec = fun_f(x);
g_vec = fun_g(x);
time_vec = 0;
[acc_vec, p_vec] = acc_p_rule(x);
while iter <= maxiter
    iter = iter+1;
    % Compute phi
    grad_f_x = grad_f(x);
    grad_g_x = grad_g(x);
    phi = min(alpha*fun_g(x),beta*(grad_g_x'*grad_g_x));
    weight = max((phi-grad_f_x'*grad_g_x)/(grad_g_x'*grad_g_x),0);
    v = grad_f_x+weight*grad_g_x;
    x = x-stepsize*v;
    % Projection to simplex
    x = ProjectOntoL1Ball(x,lambda);

    cpu_t1 = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(x)];
    time_vec = [time_vec;cpu_t1];

    % calculate accuracy and p-rule
    [acc, p] = acc_p_rule(x);
    p_vec = [p_vec;p];
    acc_vec = [acc_vec;acc];
    
    if mod(iter,5000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t1>maxtime
        break
    end
end

end

