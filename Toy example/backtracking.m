%-----------------------------------------------
% Backtracking Line search
%-----------------------------------------------
function alpha = backtracking(x,f,grad,dir)

rho = 0.9;
c = 1e-4;
alpha = 1;
while true
    x_new = x+alpha*dir;
    if f(x_new)<=f(x)+c*alpha*dir'*grad(x)
        break;
    else
        alpha = rho*alpha;
    end
end
end