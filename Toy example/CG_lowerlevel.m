function [last_iter , f_hist,x_hist] = CG_lowerlevel(fun_g,grad_g,x0,param)

disp('CG for the lowerlevel starts');

epsilon_g= param.epsilong;
x = x0; % Intital point

n = length(x0);
x_hist = x0';
f_hist = fun_g(x0);
iteration = 0;
maxiteration = 1e4;
while iteration <= maxiteration
    iteration = iteration+1;
    gam = 2/(iteration+2);
   %-----------------------------------------------------------------------
    % Find direction
    f=grad_g(x);
    options = optimoptions('linprog','Display','off');
    %d = linprog(f,A,b,Aeq,beq,lb,ub,options);
    dir = linprog(f,[1,1;4,6],[1;5],[],[],zeros(n,1),[],options);
    %----------------------------------------------------------------------
    if grad_g(x)'*(x-dir)<=epsilon_g
        break;
    end
    x = (1-gam)*x + gam*dir;
    f_hist = [f_hist;fun_g(x)];
    x_hist = [x_hist;x'];

end
disp('CG for the lower level is solved!');
last_iter = x;
end
