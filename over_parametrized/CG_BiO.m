function [f_vec1,g_vec1,time_vec1,x,acc_vec] = CG_BiO(fun_f,grad_f,grad_g,fun_g,TSA,param,x0)
% CG-BiO in "A Conditional Gradient-based Method for Simple Bilevel
% Optimization with Convex Lower-level Problem", 
% R. Jiang, N. Abolfazli, A. Mokhtari, E. Yazdandoost Hamedani, AISTATS2023
% The update rule is given by: 
%
% s_k = \argmin_{s\in X_k} \nabla f(x_k)'*s, where 
% X_k := \{ s \in Z: \nabla g(x_k)'*(s-x_k} \leq g(x_0)-g(x_k)\};
% x_{k+1} = (1-\gamma_k)x_k+\gamma_k s_k

% param definition
epsilon_f= param.epsilonf;
epsilon_g= param.epsilong;
lambda = param.lam;
maxiter = param.maxiter;

n = length(x0);
x = x0;
fun_g_x0 = param.fun_g_x0;
tic;
% algorithm
iter = 0;
f_vec1 = [];
g_vec1 = [];
time_vec1 = [];
acc_vec = [];
while iter <= maxiter
    iter = iter+1;
    gamma = 2/(iter+2+10);
    % Find direction
    b=[grad_g(x)'*x+fun_g_x0-fun_g(x); lambda];
    A = [grad_g(x)' -grad_g(x)'; ones(1,2*n)];
    lb=[sparse(2*n,1)];
    f=[grad_f(x)' -grad_f(x)'];
    options = optimoptions('linprog','Algorithm','dual-simplex','Display','off');
    % options = optimoptions('linprog','Algorithm','dual-simplex','Display','iter');
    vec = linprog(f,A,b,[],[],lb,[],options);
    d=vec(1:n)-vec(n+1:end);
    if grad_f(x)'*(x-d)<=epsilon_f && (fun_g(x)-fun_g_x0)<=epsilon_g/2
        break;
    end
    x = (1-gamma)*x + gamma*d;
    cpu_t1 = toc;
    f_vec1 = [f_vec1;fun_f(x)];
    g_vec1 = [g_vec1;fun_g(x)];
    time_vec1 = [time_vec1;cpu_t1];
    % test set accuracy
    [acc_vec] = [acc_vec;TSA(x)];
end

end