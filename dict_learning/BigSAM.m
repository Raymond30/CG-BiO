function [loss_vec_up,loss_vec_lo,rec_vec,time_vec] = BigSAM(A_train,A_test,X_last,D_init,X_init,Dstar,param)
% BiG-SAM in "A first order method for solving convex bilevel optimization problems",
% S. Sabach and S. Shtern, SIOPT 2017
% The update rule is given by: 
%
% y_{k+1} = \Pi_{Z}(x_k-\eta_g \nabla g(x_k)),
% z_{k+1} = x_k - \eta_f\nabla f(x_k),
% x_{k+1} = \alpha_{k+1} z_{k+1} + (1-\alpha_{k+1}) y_{k+1}
%
loss_test = @(D,X) norm(A_test-D*X,'fro')^2/2;
gD_test= @(D,X) (D*X-A_test)*X';
gX_test = @(D,X) D'*(D*X-A_test);

loss_lower = @(D) norm(A_train-D*X_last,'fro')^2/2;
gD_lower = @(D) (D*X_last-A_train)*X_last';

eta_up = param.eta_up;
eta_lo = param.eta_lo;
delta=param.delta;
gamma = param.gamma;

max_time = param.maxtime;
thres = param.thres; % threshold for recovery
p = param.p;
maxiter = param.maxiter;
[~,n_test] = size(A_test);


D = D_init;
X = X_init;
X_lo = 0*X;

loss_vec_up = zeros(maxiter,1);
loss_vec_lo = zeros(maxiter,1);
time_vec = zeros(maxiter,1);

% loss_vec_up(1) = loss_test(D,X);
% loss_0 = loss_lower(D_init);
% loss_vec_lo(1) = loss_0;
time_vec(1) = 0;

rec_vec = zeros(maxiter,1);
% rec_vec(1) = recovery(D_init,Dstar,thres);

iter = 0;
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

    % Descent w.r.t. the lower-level problem
    D_lo = D-eta_lo*gD_lo;
    % Projection w.r.t. the lower-level problem
    for col_i=1:p
        if norm(D_lo(:,col_i))>1
            D_lo(:,col_i) = D_lo(:,col_i)./norm(D_lo(:,col_i));
        end
    end

    for col_n = 1:n_test
        x = X(:,col_n);
        X_lo(:,col_n) = ProjectOntoL1Ball(x,delta);
    end

    % Descent w.r.t. the upper-level problem
    D_up = D-eta_up*gD;
    X_up = X-eta_up*gX;

    % Averaging
    alpha = min([2*gamma/iter,1]);
    D = alpha*D_up+(1-alpha)*D_lo;
    X = alpha*X_up+(1-alpha)*X_lo;


    time_vec(iter+1) = toc;


    loss_vec_up(iter) = loss_test(D_lo,X_lo);
    loss_vec_lo(iter) = loss_lower(D_lo);
    rec_vec(iter) = recovery(D_lo,Dstar,thres);
    if mod(iter,100) == 1
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
