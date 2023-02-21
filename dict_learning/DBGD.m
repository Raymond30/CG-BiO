function [loss_vec_up,loss_vec_lo,rec_vec,time_vec] = DBGD(A_train,A_test,X_last,D_init,X_init,Dstar,param)
% DBGD in "Bi-objective trade-off with dynamic barrier gradient descent",
% C. Gong, X. Liu, and Q. Liu, Neurips 2021
% The update rule is given by 
% x_{k+1} = \Pi_{Z}(x_k - \gamma_k(\nabla f (x_k)+\lambda_k \nabla g (x_k))),
% \lambda_k = \max\{ \frac{\phi(x_k)- \nabla f(x_k)'* \nabla g(x_k)}{\|\nabla g(x_k)\|^2},0 \},
% and \phi(x) = \min\{\alpha (g(x)-\hat{g}), \beta \|\nabla g (x)\|^2\}

loss_test = @(D,X) norm(A_test-D*X,'fro')^2/2;
gD_test= @(D,X) (D*X-A_test)*X';
gX_test = @(D,X) D'*(D*X-A_test);

loss_lower = @(D) norm(A_train-D*X_last,'fro')^2/2;
gD_lower = @(D) (D*X_last-A_train)*X_last';

stepsize = param.stepsize;
alpha = param.alpha;
beta = param.beta;
delta=param.delta;
max_time = param.maxtime;

thres = param.thres; % threshold for recovery
p = param.p;
maxiter = param.maxiter;
[~,n_test] = size(A_test);


D = D_init;
X = X_init;

loss_vec_up = zeros(maxiter,1);
loss_vec_lo = zeros(maxiter,1);
time_vec = zeros(maxiter,1);

loss_vec_up(1) = loss_test(D,X);
loss_0 = loss_lower(D_init);
loss_vec_lo(1) = loss_0;
time_vec(1) = 0;

rec_vec = zeros(maxiter,1);
rec_vec(1) = recovery(D_init,Dstar,thres);

iter = 1;
tic
while toc<= max_time
    iter = iter+1;
    gD = gD_test(D,X);
    gX = gX_test(D,X);
    gD_lo = gD_lower(D);

    if any(isnan(gD),'all') || any(isnan(gX),'all') || any(isnan(gD_lo),'all')
        iter = iter-1;
        break
    end

    % Compute phi
    phi = min(alpha*loss_lower(D),beta*(norm(gD_lo,"fro"))^2);
    weight = max((phi-vec(gD)'*vec(gD_lo))/(norm(gD_lo,"fro"))^2,0);
    v_D = gD+weight*gD_lo;
    D = D-stepsize*v_D;
    X = X-stepsize*gX;

    % Projection
    for col_i=1:p
        if norm(D(:,col_i))>1
            D(:,col_i) = D(:,col_i)./norm(D(:,col_i));
        end
    end

    for col_n = 1:n_test
        x = X(:,col_n);
        X(:,col_n) = ProjectOntoL1Ball(x,delta);
    end


    time_vec(iter) = toc;


    loss_vec_up(iter) = loss_test(D,X);
    loss_vec_lo(iter) = loss_lower(D);
    rec_vec(iter) = recovery(D,Dstar,thres);
    if mod(iter,1000) == 1
        fprintf('Iteration: %d\n',iter)
    end
end
loss_vec_up = loss_vec_up(1:iter);
loss_vec_lo = loss_vec_lo(1:iter);
rec_vec = rec_vec(1:iter);
time_vec = time_vec(1:iter);

end

function rec = recovery(D,Dstar,thres)
D = D./vecnorm(D);
[~,num_dict] = size(Dstar);
corr_mat = D'*Dstar;
num = sum(max(abs(corr_mat))>thres);
rec = num/num_dict;
end
