%-----------------------------------------------
% Gradient Method
%-----------------------------------------------

function [last_iter,f_hist,x_hist] = gradient_method2(proj,f,fstar,grad,x0,stepsize,accuracy)
disp('Gradient method starts');
x = x0; % Intital point
f_hist = f(x0);
x_hist = x0';
l0=0;
while true
%while @g(x)-gstar<=epsilon/4
    dir = -grad(x);
    x_old = x;
    step = stepsize(x,f,grad,dir);
    x = x+step*dir; % main stepeak;

    x = proj(x);
    
    f_hist = [f_hist;f(x)];
    x_hist = [x_hist;x'];
    if f(x)-fstar<=accuracy
        break;
    end
end 
disp('desired accuracy for Gradient method is achived!');
last_iter = x;
end