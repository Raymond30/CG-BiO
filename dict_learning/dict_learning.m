function [time_vec,time_vec_proj,time_vec2,time_vec_sam,...
   rec_vec,rec_vec_proj,rec_vec2,rec_vec_sam,...
   loss_vec_up,loss_vec_up_proj,loss_vec_up2,loss_vec_up_sam,...
   loss_vec_lo,loss_vec_lo_proj,loss_vec_lo2,loss_vec_lo_sam]=dict_learning(seed,figs)
% generate the data
p = 50; % number of atoms
m = 25; % dimension of data
n_old = 250; % number of lower-level samples
n_new = 200; % number of upper-level samples
thres = 0.9; % threshold for recovery rate
sigma = 0.01; % noise level
max_time = 39;  % max running time

% seed = 2020;
rng(seed)
% figs = true;

Dstar = randn(m,p); % true dictionary
Dstar = Dstar./vecnorm(Dstar); % normalize the atoms 
k_spar = 5; % number of nonzeros in each coefficient vector
mask = zeros(p,n_old);
for i=1:n_old
    mask(randsample(round(4*p/5),k_spar),i)=(0.8*rand(k_spar,1)+0.2).*(2*randi([0 1],k_spar,1)-1);
end
X_old = mask; % true coefficient matrix for the old dataset
A_old = Dstar*X_old+sigma*randn(m,n_old);

mask = zeros(p,n_new);
for i=1:n_new
    mask(randsample(round(3*p/5):p,k_spar),i)=(0.8*rand(k_spar,1)+0.2).*(2*randi([0 1],k_spar,1)-1);
end
X_new = mask; % true coefficient matrix for the new dataset
A_new = Dstar*X_new+sigma*randn(m,n_new);


%% solving the lower-level problem initially by CG
epsilon_g = 1e-6;
maxiter = 1e4;
delta = 3;

param.delta = delta;
param.thres = thres;
param.p = round(4*p/5);
param.eps = epsilon_g;
param.eps2 = 1e-10;
param.maxiter = maxiter;

disp('Initialization starts!');
[D_last,X_last,~,~] = CG_init(A_old,Dstar,param);
disp('Initialization done!');

p_res = p-round(4*p/5);
D_last = [D_last,zeros(m,p_res)];
X_last = [X_last;zeros(p_res,n_old)];
loss_0 = norm(A_old-D_last*X_last,'fro')^2/2;
%% CG-BiO

gamma0 = .3;
maxiter_up = 5e4;

param.delta = delta;
param.maxtime = max_time;
param.gamma0 = gamma0; % initial stepsize
param.thres = thres; % threshold for recovery
param.p = p;
param.maxiter = maxiter_up;

X_up_init = randn(p,n_new);
X_up_init = X_up_init./vecnorm(X_up_init,1)*delta;
[loss_vec_up,loss_vec_lo,rec_vec,time_vec] = CG_BiO(A_old,A_new,X_last,D_last,X_up_init,Dstar,param);
% normalize the loss
loss_vec_up = loss_vec_up/n_new;
loss_vec_lo = (loss_vec_lo-loss_0)/n_old;
%% The same stepsize but without cutting plane
% X_up_init = zeros(p,n_new);
[loss_vec_up2,loss_vec_lo2,rec_vec2,time_vec2] = CG_upper(A_old,A_new,X_last,D_last,X_up_init,Dstar,param);
loss_vec_up2 = loss_vec_up2/n_new;
loss_vec_lo2 = (loss_vec_lo2-loss_0)/n_old;
%% Yousefian's algorithm
disp('a-IRG algorithm starts!')
[loss_vec_up_proj,loss_vec_lo_proj,rec_vec_proj,time_vec_proj] = Alg_projection(A_old,X_last,A_new,Dstar,...
    D_last,X_up_init,param);
disp('a-IRG algorithm done!');

loss_vec_up_proj = loss_vec_up_proj/n_new;
loss_vec_lo_proj = (loss_vec_lo_proj-loss_0)/n_old;

%% BiG-SAM
eta_up_list = [1e-2,1e-1,1];
eta_lo_list = [1e-2,1e-1,1];
gamma_list = [1e-1,1,10];

loss_list = zeros(3,3,3);

for idx_up = 1:3
    for idx_lo = 1:3
        for idx_gamma = 1:3
            param.eta_up = eta_up_list(idx_up);
            param.eta_lo = eta_lo_list(idx_lo);
            param.gamma =  gamma_list(idx_gamma);
            param.maxtime = 10;
            disp('BiG-SAM algorithm starts!')
            [loss_vec_up_sam,loss_vec_lo_sam,~,~] = BigSAM(A_old,A_new,X_last,...
            D_last,X_up_init,Dstar,param);
            disp('BiG-SAM algorithm done!');
            loss_list(idx_up,idx_lo,idx_gamma) = loss_vec_up_sam(end)/n_new+(loss_vec_lo_sam(end)-loss_0)/n_old;
        end
    end
end
%%
[~,I] = min(loss_list,[],'all','linear');
[I1,I2,I3] = ind2sub([3,3,3],I);
param.eta_up = eta_up_list(I1);
param.eta_lo = eta_lo_list(I2);
param.gamma =  gamma_list(I3);

% param.eta_up = 0.1;
% param.eta_lo = 0.1;
% param.gamma = 10;

param.maxtime = 39;
disp('BiG-SAM algorithm starts!')
[loss_vec_up_sam,loss_vec_lo_sam,rec_vec_sam,time_vec_sam] = BigSAM(A_old,A_new,X_last,...
    D_last,X_up_init,Dstar,param);
disp('BiG-SAM algorithm done!');

loss_vec_up_sam = loss_vec_up_sam/n_new;
loss_vec_lo_sam = (loss_vec_lo_sam-loss_0)/n_old;


%% Figures
if figs == true
    figure;
    set(0,'defaulttextinterpreter','latex')
    set(gcf,'DefaultLineLinewidth',5)
    set(gcf,'DefaultLineMarkerSize',16);
    set(gcf,'Position',[331,215,720,538])
    % set(gcf,'WindowState','maximized');
    N_marker = 10;
    time_idx = linspace(0,max_time,N_marker);
    marker_idx = zeros(N_marker,1);
    marker_idx_proj = zeros(N_marker,1);
    marker_idx2 = zeros(N_marker,1);
    marker_idx_sam = zeros(N_marker,1);
    marker_idx_isam = zeros(N_marker,1);
    for j=1:N_marker
        [~,idx] = min(abs(time_vec-time_idx(j)));
        marker_idx(j) = idx;
        [~,idx] = min(abs(time_vec_proj-time_idx(j)));
        marker_idx_proj(j) = idx;
        [~,idx] = min(abs(time_vec2-time_idx(j)));
        marker_idx2(j) = idx;
        [~,idx] = min(abs(time_vec_sam-time_idx(j)));
        marker_idx_sam(j) = idx;
    end
    plot(time_vec,rec_vec,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx)
    hold on
    plot(time_vec_sam,rec_vec_sam,'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx_sam)
    plot(time_vec_proj, rec_vec_proj,'s-','DisplayName','a-IRG','MarkerIndices', marker_idx_proj)
    plot(time_vec2,rec_vec2,'d-','DisplayName','FW (w/o cutting plane)','MarkerIndices', marker_idx2)


    lgd = legend();
    lgd.Position=[0.521319094763862 0.416352230128746 0.37569514380561 0.252044610374479];
    ylabel('Recovery rate')
    xlabel('time (s)')
    set(gca,'FontSize',24);
    set(gca,'YLim',[0,1])
    legend('Interpreter','latex')
    grid on;
    pbaspect([1 0.7 1])
    % print('-depsc2','-r600','./figs/recovery_new.eps')
    %% Upper-level objective
    figure;
    set(0,'defaulttextinterpreter','latex')
    set(gcf,'DefaultLineLinewidth',5)
    set(gcf,'DefaultLineMarkerSize',16);
    set(gcf,'Position',[331,215,720,538])
    % set(gcf,'WindowState','maximized');
    semilogy(time_vec,loss_vec_up,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx)
    hold on
    semilogy(time_vec_sam,loss_vec_up_sam,'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx_sam)
    semilogy(time_vec_proj,loss_vec_up_proj,'s-','DisplayName','a-IRG','MarkerIndices', marker_idx_proj)
    semilogy(time_vec2,loss_vec_up2,'d-','DisplayName','FW (w/o cutting plane)','MarkerIndices', marker_idx2)


    legend
    
    ylabel('$f(\tilde{\mathbf{D}}_k,\tilde{\mathbf{X}}_k)$')
    xlabel('time (s)')
    set(gca,'FontSize',24);
    set(gca,'YLim',[1e-6,1])
    legend('Interpreter','latex','Location','southwest')
    grid on;
    pbaspect([1 0.7 1])
    % print('-depsc2','-r600','./figs/nonconvex_upper_new.eps')
    %% Lower-level objective
    figure;
    set(0,'defaulttextinterpreter','latex')
    set(gcf,'DefaultLineLinewidth',5)
    set(gcf,'DefaultLineMarkerSize',16);
    set(gcf,'Position',[331,215,720,538])
    % set(gcf,'WindowState','maximized');
    
    semilogy(time_vec, loss_vec_lo,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx)
    hold on
    semilogy(time_vec_sam, loss_vec_lo_sam,'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx_sam)
    semilogy(time_vec_proj, loss_vec_lo_proj,'s-','DisplayName','a-IRG','MarkerIndices', marker_idx_proj)
    semilogy(time_vec2, loss_vec_lo2,'d-','DisplayName','FW (w/o cutting plane)','MarkerIndices', marker_idx2)

    
    legend
    
    ylabel('$g(\tilde{\mathbf{D}}_k)-g(\tilde{\mathbf{D}}_0)$')
    xlabel('time (s)')
    set(gca,'FontSize',24);
    set(gca,'YLim',[1e-22,1])
    legend('Interpreter','latex','Location','southwest')
    grid on;
    pbaspect([1 0.7 1])
    % print('-depsc2','-r600','./figs/nonconvex_lower_new.eps')
    
end
