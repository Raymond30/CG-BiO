function [last_iter , g_hist] = CG_lowerlevel(fun_g,grad_g,x0,param)
% Standard CG algorithm for the lower-level problem
disp('CG for the lower level starts');

epsilon_g= param.epsilong;
lambda1 = param.lam1;
tau = param.tau;
eta = param.eta;
x = x0; % Intital point

g_hist = fun_g(x0);
fun_g_x = g_hist;

iteration = 0;
maxiteration = param.maxiter;

while iteration <= maxiteration
    iteration = iteration+1;

    grad_x = grad_g(x);
    s = linear_l1(grad_x,lambda1);
    d = s-x; gap = -grad_x'*d;

    if gap<=epsilon_g
        break;
    end

    % Initial trial stepsize
    if iteration == 1
        M = norm(grad_g(x)-grad_g(x+1e-4*d))/(1e-4*norm(d));
    else 
        M = eta*M;
    end
    gam = min(gap/(M*(d'*d)),1);

    x_new = x+gam*d;
    fun_g_x_new = fun_g(x_new);

    % Line search
    while fun_g_x_new>fun_g_x-gam*gap+gam^2*M*(d'*d)/2
        M = tau*M;
        gam = min(gap/(M*(d'*d)),1);
        x_new = x+gam*d;
        fun_g_x_new = fun_g(x_new);
    end
    x = x_new;
    g_hist = [g_hist;fun_g_x_new];
    fun_g_x = fun_g_x_new;
end
disp('CG for the lower level is solved!');
last_iter = x;
end

function x = linear_l1(c,lambda)
% find x to minimize c'*x  s.t.  norm(x,1)<=lambda
x = sparse(length(c),1);
[~,ind] = max(abs(c));
x(ind) = -sign(c(ind))*lambda;
end
