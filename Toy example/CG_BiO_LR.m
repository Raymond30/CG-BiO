%---------------------Bi-Level Frank-Wolfe Algorithm------------------------
%--------------------------------------------------------------------------

function [f_vec1,g_vec1,x_hist] = CG_BiO_LR(fun_f,grad_f,grad_g,fun_g,param,x0)
disp('CG-BiO Algorithm starts');
tic;
%param definition
epsilon_f= param.epsilonf;
epsilon_g= param.epsilong;

maxiter = 1e4;

n = length(x0);
x = x0;
fun_g_x0 = fun_g(x0);

%algorithm
iter = 0;
f_vec1 = [];
g_vec1 = [];
x_hist = x0';
%maxiter = 1e5;
while iter <= maxiter
    iter = iter+1;
    gamma = 2/(iter+2); %min(epsilon_f/(lip_f*B^2), epsilon_g/(lip_g*B^2));
   %-----------------------------------------------------------------------
    % Find direction
    f=grad_f(x);
    options = optimoptions('linprog','Display','off');
    d = linprog(f,[1,1;4,6;grad_g(x)'],[1;5;grad_g(x)'*x+fun_g_x0-fun_g(x)],[],[],zeros(n,1),[],options);

    if grad_f(x)'*(x-d)<=epsilon_f && grad_g(x)'*(x-d)<=epsilon_g
        break;
    end
    x = (1-gamma)*x + gamma*d;
    cpu_t1 = toc;
    f_vec1 = [f_vec1;fun_f(x)];
    %gradg_vec1 = [gradg_vec1;norm(grad_g(x))];
    g_vec1 = [g_vec1;fun_g(x)];
    x_hist = [x_hist;x'];
    % calculate TSA for each iterate point
    
%     pr1_b0 = 1 ./ (1+exp(A3* x));
%     b_test1 = (pr1_b0 <=  0.5);
%     N1 = sum(b_test1*2-1 == b3);
%     acc_vec = [acc_vec;(N1 / size(b3,1))*100];
end

end