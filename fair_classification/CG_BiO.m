function [f_vec1,g_vec1,time_vec1,x,acc_vec,p_vec] = CG_BiO(fun_f,grad_f,grad_g,fun_g,acc_p_rule,param,x0)
% CG-BiO in "A Conditional Gradient-based Method for Simple Bilevel
% Optimization with Convex Lower-level Problem", 
% R. Jiang, N. Abolfazli, A. Mokhtari, E. Yazdandoost Hamedani, AISTATS2023
% The update rule is given by: 
%
% s_k = \argmin_{s\in X_k} \nabla f(x_k)'*s, where 
% X_k := \{ s \in Z: \nabla g(x_k)'*(s-x_k} \leq g(x_0)-g(x_k)\};
% x_{k+1} = (1-\gamma_k)x_k+\gamma_k s_k

%param definition
epsilon_f= param.epsilonf;
epsilon_g= param.epsilong;
lambda = param.lam;
maxiter = param.maxiter;
gamma0 = param.gamma;

% Initialization
n = length(x0);
x = x0;
fun_g_x0 = param.fun_g_x0;
iter = 0;
f_vec1 = fun_f(x);
g_vec1 = fun_g_x0;
time_vec1 = 0;
[acc_vec,p_vec] = acc_p_rule(x);
fun_g_x = fun_g_x0;
tic;
while iter <= maxiter
    iter = iter+1;
    gamma = gamma0/sqrt(iter);

    % Solve the LP to find the direction
    grad_g_x = grad_g(x);
    b=[grad_g_x'*x+fun_g_x0-fun_g_x; lambda];
    A = [grad_g_x' -grad_g_x'; ones(1,2*n)];
    lb=[sparse(2*n,1)];
    grad_f_x = grad_f(x);
    f=[grad_f_x' -grad_f_x'];
    options = optimoptions('linprog','Algorithm','dual-simplex','Display','off');
    vec = linprog(f,A,b,[],[],lb,[],options);
    s=vec(1:n)-vec(n+1:end);

    if grad_f_x'*(x-s)<=epsilon_f && (fun_g_x-fun_g_x0)<=epsilon_g/2
        break;
    end
    % update
    x = (1-gamma)*x + gamma*s;
    cpu_t1 = toc;
    f_vec1 = [f_vec1;fun_f(x)];
    fun_g_x = fun_g(x);
    g_vec1 = [g_vec1;fun_g_x];
    time_vec1 = [time_vec1;cpu_t1];

    % calculate the p-rule and the accuracy for each iterate point
    [acc,p] = acc_p_rule(x);
    p_vec = [p_vec;p];
    acc_vec = [acc_vec;acc];
end

end