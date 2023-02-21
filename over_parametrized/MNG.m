function [f_vec1,g_vec1,time_vec1,x,acc_vec] = MNG(A,b,fun_g,grad_g,TSA,param,x0)
% MNG in "A first order method for finding minimal norm-like solutions of convex optimization problems",
% A. Beck and S. Sabach, Math. Program. 2014
% The update rule is given by: 
% x_{k+1} = \argmin_{ x\in Q_k \cap W_k} f(x),
% s.t. \langle G_M(x_k), x_k-x \rangle \geq \frac{3}{4M}\|G_M(x_k)\|^2, 
% \langle \grad f(x_k), z-x_k \rangle \geq 0, 
%
% where G_M(x) = M[\bx-\Pi_{Z}(x-\frac{1}{M}\grad g(x))]

disp('Mininum norm gradient Algorithm starts');
% param definition
M = param.M; 
lambda = param.lam;

maxiter = param.maxiter;
maxtime = param.maxtime;


fun_f= @(x) sum_square(A*x-b)/2;
grad_f= @(x) A'*(A*x-b);

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
    y = ProjectOntoL1Ball(x-1/M*grad_g(x),lambda);
    G_M = M*(x-y);
    A_ineq = [G_M';-grad_f(x)'];
    b_ineq = [G_M'*x-3/4/M*(G_M'*G_M);-grad_f(x)'*x];
    H = full(A'*A);
    f = -A'*b;
    options = optimoptions('quadprog','Display','none');
    x = quadprog(H,f,A_ineq,b_ineq,[],[],[],[],[],options);
    
    cpu_t1 = toc;
    f_vec1 = [f_vec1;fun_f(x)];
    g_vec1 = [g_vec1;fun_g(y)];
    time_vec1 = [time_vec1;cpu_t1];
    % test set accuracy
    [acc_vec] = [acc_vec;TSA(x)];
    
    if mod(iter,100) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t1>maxtime
        break
    end
end
disp('Mininum norm gradient Algorithm ends');
end