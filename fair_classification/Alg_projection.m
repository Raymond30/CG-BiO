function[loss_vec_f,loss_vec_g,time_vec,x,acc_vec,p_vec] = Alg_projection(fun_f,grad_f,grad_g,fun_g,acc_p_rule,param,x0)
% a-IRG in "A method with convergence rates for optimization
% problems with variational inequality constraints", H. D. Kaushik and F. Yousefian, SIOPT 2021
% The update rule is given by: 
% 
% x_{k+1} = \Pi_{\mathcal{Z}}(x_k - \gamma_k(\nabla g(x_k)+\eta_k \nabla f(x_k)))
% 
% We choose \gamma_k = \gamma_0/sqrt{k+1} and \eta_k = \eta_0/(k+1)^0.25

% parameters
lambda=param.lam;
eta_0 = param.eta;
gamma_0 = param.gamma;
maxiter = param.maxiter;
maxtime = param.maxtime;

% initialize
x = x0;
loss_vec_f = zeros(maxiter+1,1);
loss_vec_g = zeros(maxiter+1,1);
time_vec = zeros(maxiter+1,1);
p_vec =  zeros(maxiter+1,1);   
acc_vec = zeros(maxiter+1,1);
loss_vec_f(1) = fun_f(x);
loss_vec_g(1) = fun_g(x);
time_vec(1) = 0;

[acc,p] = acc_p_rule(x);
p_vec(1) = p;
acc_vec(1) = acc;

k=0;
tic;
while toc<maxtime
    k = k+1;
    eta_k = (eta_0)/(k)^0.25;
    gamma_k = gamma_0/sqrt(k);
    % Descent step
    x = x - gamma_k*(grad_g(x)+eta_k*(grad_f(x)));
    % Projection step
    x = ProjectOntoL1Ball(x,lambda);
    
    time_vec(k+1) = toc;
    loss_vec_f(k+1) = fun_f(x); 
    loss_vec_g(k+1) = fun_g(x);
    
    % p%-rule and test accuracy
    [acc,p] = acc_p_rule(x);
    p_vec(k+1) = p;
    acc_vec(k+1) = acc;    
end

time_vec = time_vec(1:k+1);
loss_vec_f = loss_vec_f(1:k+1);
loss_vec_g = loss_vec_g(1:k+1);
p_vec = p_vec(1:k+1);
acc_vec = acc_vec(1:k+1);
end

