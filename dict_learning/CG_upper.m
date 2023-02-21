function [loss_vec_up2,loss_vec_lo2,rec_vec2,time_vec2] = CG_upper(A_train,A_test,X_last,D_up_init,X_up_init,Dstar,param)
% The baseline method that performs CG on the upper-level objective
loss_test = @(D,X) norm(A_test-D*X,'fro')^2/2;
gD_test= @(D,X) (D*X-A_test)*X';
gX_test = @(D,X) D'*(D*X-A_test);

loss_lower = @(D) norm(A_train-D*X_last,'fro')^2/2;

delta = param.delta;
max_time = param.maxtime;
gamma0 = param.gamma0; % initial stepsize
thres = param.thres; % threshold for recovery
p = param.p;
maxiter = param.maxiter;
[~,n_test] = size(A_test);

D_up = D_up_init;
X_up = X_up_init;

loss_vec_up2 = zeros(maxiter+1,1);
loss_vec_lo2 = zeros(maxiter+1,1);
time_vec2 = zeros(maxiter+1,1);

loss_vec_up2(1) = loss_test(D_up,X_up);
loss_vec_lo2(1) = loss_lower(D_up);
time_vec2(1) = 0;

rec_vec2 = zeros(maxiter+1,1);
rec_vec2(1) = recovery(D_up_init,Dstar,thres);

iter = 0;
tic;
while toc<=max_time
    iter = iter+1;
    gamma = gamma0/sqrt(iter);
    gD = gD_test(D_up,X_up);
    gX = gX_test(D_up,X_up);
    % solve the subproblem
    D_atom = -gD./(vecnorm(gD)+eps);

    D_up = (1-gamma)*D_up+gamma*D_atom;

    [~,max_idx] = max(abs(gX));
    X_atom = zeros(p,n_test);
    for j=1:n_test
        X_atom(max_idx(j),j) = -delta*sign(gX(max_idx(j),j));
    end
    X_up = (1-gamma)*X_up+gamma*X_atom;
    time_vec2(iter+1) = toc;

    loss_vec_up2(iter+1) = loss_test(D_up,X_up);
    loss_vec_lo2(iter+1) = loss_lower(D_up);
    rec_vec2(iter+1) = recovery(D_up,Dstar,thres);
    if mod(iter,1000) == 1
        fprintf('Iteration: %d\n',iter)
    end
end

loss_vec_up2 = loss_vec_up2(1:iter+1);
loss_vec_lo2 = loss_vec_lo2(1:iter+1);
rec_vec2 = rec_vec2(1:iter+1);
time_vec2 = time_vec2(1:iter+1);
end

function rec = recovery(D,Dstar,thres)
D = D./vecnorm(D);
[~,num_dict] = size(Dstar);
corr_mat = D'*Dstar;
num = sum(max(abs(corr_mat))>thres);
rec = num/num_dict;
end