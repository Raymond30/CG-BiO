function[loss_vec_f,loss_vec_g,rec_vec,time_vec] = Alg_projection(A_train,X_train,A_test,Dstar,...
    D0,X0,param)
% a-IRG in "A method with convergence rates for optimization
% problems with variational inequality constraints", H. D. Kaushik and F. Yousefian, SIOPT 2021
% The update rule is given by: 
% 
% x_{k+1} = \Pi_{\mathcal{Z}}(x_k - \gamma_k(\nabla g(x_k)+\eta_k \nabla f(x_k)))
% 
% We choose \gamma_k = \gamma_0/sqrt{k+1} and \eta_k = \eta_0/(k+1)^0.25

% get dimension
[p,~] = size(X_train);
[~,n_test] = size(A_test);

% Define loss and gradients
loss_f = @(D,X) norm(A_test-D*X,'fro')^2/2;
loss_g = @(D) norm(A_train-D*X_train,'fro')^2/2;

grad_f_D = @(D,X) (D*X-A_test)*X';
grad_f_X = @(D,X) D'*(D*X-A_test);
grad_g_D = @(D) (D*X_train-A_train)*X_train'; 

eta_0 = 1;
gamma_0 = 0.01;

delta=param.delta;
thres=param.thres;

% Initialization
D = D0;
X = X0;
%% algorithm
maxiter = param.maxiter;
maxtime = param.maxtime;

loss_vec_f = zeros(maxiter+1,1);
loss_vec_g = zeros(maxiter+1,1);
time_vec = zeros(maxiter+1,1);
loss_vec_f(1) = loss_f(D,X);
loss_vec_g(1) = loss_g(D);
time_vec(1) = 0;

rec_vec = zeros(maxiter+1,1);
rec_vec(1) = recovery(D,Dstar,thres);

k=0;
tic;
while toc<maxtime
    k = k+1;
    eta_k = (eta_0)/(k)^0.25;
    gamma_k = gamma_0/sqrt(k);
    % Descent step
    D = D - gamma_k*(grad_g_D(D)+eta_k*(grad_f_D(D,X)));
    X = X - gamma_k*(eta_k*(grad_f_X(D,X)));
    
    % Projection step
    for col_i=1:p
        if norm(D(:,col_i))>1
            D(:,col_i) = D(:,col_i)./norm(D(:,col_i));
        end
    end
    for col_n = 1:n_test
        if norm(X(:,col_n),1)>delta
            x = X(:,col_n);
            X(:,col_n) = ProjectOntoL1Ball(x,delta);
        end
    end

    time_vec(k+1) = toc;
    loss_vec_f(k+1) = loss_f(D,X); 
    loss_vec_g(k+1) = loss_g(D);
    rec_vec(k+1) = recovery(D,Dstar,thres);
end

time_vec = time_vec(1:k+1);
loss_vec_f = loss_vec_f(1:k+1);
loss_vec_g = loss_vec_g(1:k+1);
rec_vec = rec_vec(1:k+1);
end

function rec = recovery(D,Dstar,thres)
D = D./vecnorm(D);
[~,num_dict] = size(Dstar);
corr_mat = D'*Dstar;
num = sum(max(abs(corr_mat))>thres);
rec = num/num_dict;
end