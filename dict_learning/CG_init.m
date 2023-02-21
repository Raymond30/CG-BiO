function [D,X,loss_vec,rec_vec] = CG_init(A_train,Dstar,param)
% CG on the lower-level problem
% A_train: the dataset in the lower level
% Dstar: the true dictionary
% m: dimension of data
% p: number of atoms
% delta: the l1-norm constraint
% epsilon: prescribed accuracy
% maxiter: the max. number of iterations
[m,n_train] = size(A_train);
p = param.p;
delta = param.delta;
epsilon = param.eps;
epsilon2 = param.eps2;
maxiter = param.maxiter;
thres = param.thres;

D_init = randn(m,p);
D_init = D_init./vecnorm(D_init);
X_init = zeros(p,n_train);
loss_train = @(D,X) norm(A_train-D*X,'fro')^2/2;
gD_train= @(D,X) (D*X-A_train)*X';
gX_train = @(D,X) D'*(D*X-A_train);

loss_vec1 = zeros(maxiter+1,1);
loss_vec1(1) = loss_train(D_init,X_init);
rec_vec1 = zeros(maxiter+1,1);
rec_vec1(1) = recovery(D_init,Dstar,thres);

D = D_init;
X = X_init;

for iter = 1:maxiter
    eta = 1/sqrt(iter);
    gD = gD_train(D,X);
    gX = gX_train(D,X);
    D_atom = -gD./(vecnorm(gD)+eps);

    [~,max_idx] = max(abs(gX));
    X_atom = zeros(p,n_train);
    for j=1:n_train
        X_atom(max_idx(j),j) = -delta*sign(gX(max_idx(j),j));
    end
    
    FW_gap = trace(gD'*(D-D_atom))+trace(gX'*(X-X_atom));

    D = (1-eta)*D+eta*D_atom;
    X = (1-eta)*X+eta*X_atom;

    loss_vec1(iter+1) = loss_train(D,X);
    rec_vec1(iter+1) = recovery(D,Dstar,thres);

    if FW_gap<epsilon
        break
    end
end
loss_vec1 = loss_vec1(1:iter+1);
rec_vec1 = rec_vec1(1:iter+1);


% We fix X and further run FW on D
loss_vec2 = zeros(maxiter,1);
rec_vec2 = zeros(maxiter,1);

for iter = 1:maxiter
    gD = gD_train(D,X);
    D_atom = -gD./(vecnorm(gD)+eps);
    D_dir = D-D_atom;
    eta = trace(gD'*D_dir)/(trace(X'*(D_dir'*D_dir)*X));
    eta = min([eta,1]);
    
    FW_gap = trace(gD'*(D-D_atom));
    D = (1-eta)*D+eta*D_atom;
    loss_vec2(iter) = loss_train(D,X);
    rec_vec2(iter) = recovery(D,Dstar,thres);
    if FW_gap<epsilon2
        break
    end
end
loss_vec2 = loss_vec2(1:iter);
rec_vec2 = rec_vec2(1:iter);
loss_vec = [loss_vec1;loss_vec2];
rec_vec = [rec_vec1;rec_vec2];
end

function rec = recovery(D,Dstar,thres)
D = D./vecnorm(D);
[~,num_dict] = size(Dstar);
corr_mat = D'*Dstar;
num = sum(max(abs(corr_mat))>thres);
rec = num/num_dict;
end