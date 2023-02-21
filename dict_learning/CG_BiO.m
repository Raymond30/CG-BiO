function [loss_vec_up,loss_vec_lo,rec_vec,time_vec] = CG_BiO(A_train,A_test,X_last,D_up_init,X_up_init,Dstar,param)
% CG-BiO in "A Conditional Gradient-based Method for Simple Bilevel
% Optimization with Convex Lower-level Problem", 
% R. Jiang, N. Abolfazli, A. Mokhtari, E. Yazdandoost Hamedani, AISTATS2023
% The update rule is given by: 
%
% s_k = \argmin_{s\in X_k} \nabla f(x_k)'*s, where 
% X_k := \{ s \in Z: \nabla g(x_k)'*(s-x_k} \leq g(x_0)-g(x_k)\};
% x_{k+1} = (1-\gamma_k)x_k+\gamma_k s_k

loss_test = @(D,X) norm(A_test-D*X,'fro')^2/2;
gD_test= @(D,X) (D*X-A_test)*X';
gX_test = @(D,X) D'*(D*X-A_test);

loss_lower = @(D) norm(A_train-D*X_last,'fro')^2/2;
gD_lower = @(D) (D*X_last-A_train)*X_last';

delta = param.delta;
max_time = param.maxtime;
gamma0 = param.gamma0; % initial stepsize
thres = param.thres; % threshold for recovery
p = param.p;
maxiter = param.maxiter;
[~,n_test] = size(A_test);


D_up = D_up_init;
X_up = X_up_init;

loss_vec_up = zeros(maxiter+1,1);
loss_vec_lo = zeros(maxiter+1,1);
time_vec = zeros(maxiter+1,1);

loss_vec_up(1) = loss_test(D_up,X_up);
loss_0 = loss_lower(D_up_init);
loss_vec_lo(1) = loss_0;
time_vec(1) = 0;

rec_vec = zeros(maxiter+1,1);
rec_vec(1) = recovery(D_up_init,Dstar,thres);

iter = 0;
tic
while toc<= max_time
    iter = iter+1;
    
    gamma = gamma0/sqrt(iter);
    gD = gD_test(D_up,X_up);
    gX = gX_test(D_up,X_up);
    gD_lo_cur = gD_lower(D_up);
    loss_lo_cur = loss_lower(D_up);
    Delta = loss_0-loss_lo_cur+trace(gD_lo_cur'*D_up);

    D_atom = FW_sub(gD,gD_lo_cur,Delta);
    D_up = (1-gamma)*D_up+gamma*D_atom;

    
    
    [~,max_idx] = max(abs(gX));
    X_atom = zeros(p,n_test);
    for j=1:n_test
        X_atom(max_idx(j),j) = -delta*sign(gX(max_idx(j),j));
    end
    X_up = (1-gamma)*X_up+gamma*X_atom;
    time_vec(iter+1) = toc;

    loss_vec_up(iter+1) = loss_test(D_up,X_up);
    loss_vec_lo(iter+1) = loss_lower(D_up);
    rec_vec(iter+1) = recovery(D_up,Dstar,thres);
    if mod(iter,100) == 1
        fprintf('Iteration: %d\n',iter)
    end
end
loss_vec_up = loss_vec_up(1:iter+1);
loss_vec_lo = loss_vec_lo(1:iter+1);
rec_vec = rec_vec(1:iter+1);
time_vec = time_vec(1:iter+1);

end

function D_atom = FW_sub(G1,G2,Delta)
% min trace(G1'*D)
% s.t. trace(G2'*D)<=Delta
% ||D_i||_2<=1
D = -G1./(vecnorm(G1)+eps);
if trace(G2'*D)<=Delta
    D_atom = D;
    return
end
D_lambda = @(lambda) -(G1+lambda*G2)./(vecnorm(G1+lambda*G2)+eps);
eqn_lambda = @(lambda) trace(G2'*D_lambda(lambda))-Delta;
lambda = fzero(eqn_lambda, 0);
D_atom = D_lambda(lambda);
end

function rec = recovery(D,Dstar,thres)
D = D./vecnorm(D);
[~,num_dict] = size(Dstar);
corr_mat = D'*Dstar;
num = sum(max(abs(corr_mat))>thres);
rec = num/num_dict;
end