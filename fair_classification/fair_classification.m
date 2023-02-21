clear;
%% load data
seed = 123456;

rng(seed);
load("adult.mat")
% X: the data matrix
% x_control: the sensitive attribute
% y: the target label

% standardize the features
X = (X-min(X,[],1))./(max(X,[],1)-min(X,[],1));
x_control = x_control';
y = y';

% append the intercept
[m,n]= size(X);
X = [X, ones(m,1)];
n = n+1;

trp=0.2; % 2,000 datapoints as the training set
tstp = 0.1; % 1,000 datapoints as the test set
idx=randperm(m); % randomly permute the dataset

% training dataset
m_train = round(trp*m);
idx_train = idx(1:m_train);
X_train = X(idx_train,:);
x_control_train = x_control(idx_train);
y_train = y(idx_train);

% test dataset 
m_test = round(tstp*m);
idx_test = idx(m_train+1:m_train+m_test);
X_test = X(idx_test,:);
x_control_test = x_control(idx_test);
y_test = y(idx_test);

% problem parameter
lambda=100;
%% function definition
x_control_train_centered = x_control_train-mean(x_control_train);

cov = @(x) (x_control_train_centered'*logistic(X_train*x))/m_train;

% upper-level objective
fun_f = @(x) cov(x)^2;
grad_f = @(x) 2*cov(x)*X_train'*(x_control_train_centered.*exp(-sign(X_train*x).*(X_train*x))./(1+exp(-sign(X_train*x).*(X_train*x)).^2))/m_train;

% lower-level objective
fun_g = @(x) -sum(log_logistic(y_train.*X_train*x))/m_train;
grad_g = @(x) X_train'*(-y_train.*logistic(-y_train.*X_train*x))/m_train;


%% Finding optimal solution
maxiter = 1e4;
options = optimoptions('fmincon','Algorithm','interior-point','MaxFunctionEvaluations',maxiter,'OptimalityTolerance',1e-10,'SpecifyObjectiveGradient',true);
obj_logistic = @(x) deal(fun_g(x(1:n)-x(n+1:end)),[grad_g(x(1:n)-x(n+1:end));-grad_g(x(1:n)-x(n+1:end))]);

% We solve the lower-level problem with two random initial points
x0 = randn(n,1);
if norm(x0,1) > lambda
    x0 = lambda/2*x0/norm(x0,1);
end
x0 = [max(x0,0);-min(x0,0)];
A = ones(1,2*n);
b = lambda;
[xstar_g,gstar] = fmincon(obj_logistic,x0,A,b,[],[],zeros(2*n,1),[],[],options);

x0 = randn(n,1);
if norm(x0,1) > lambda
    x0 = lambda/2*x0/norm(x0,1);
end
x0 = [max(x0,0);-min(x0,0)];
[xstar_g_,gstar_] = fmincon(obj_logistic,x0,A,b,[],[],zeros(2*n,1),[],[],options);

% xstar_g differs from xstar_g_, suggesting that there are multiple optimal
% solutions
norm(xstar_g_-xstar_g,1)
%% CG for the lower-level subproblem
epsilon_f = 1e-4;
epsilon_g = 1e-4;

x_init = sparse(n,1);
param.epsilong = epsilon_g/2;
param.lam1=lambda;
param.maxiter=1e4;
% linesearch parameters
param.tau = 2;
param.eta = 0.9;
tic;
[last_iter, f_hist] = CG_lowerlevel(fun_g,grad_g,x_init,param);
time_init = toc;

%% CG-BiO algorithm
param.epsilonf = epsilon_f;
param.epsilong = epsilon_g;
param.lam=lambda;
param.fun_g_x0 = fun_g(last_iter);
param.maxiter=5e3;

gamma = 5e-3;
param.gamma = gamma;
    
disp('CG-BiO Algorithm starts');
[f_vec1,g_vec1,time_vec1,x,tsa_BFW,p_BFW] = CG_BiO(fun_f,grad_f,grad_g,fun_g,...
    @(x) acc_p_rule(X_test,y_test,x_control_test,x), param,last_iter);
disp('CG-BiO Solution Achieved!');

time_vec1 = time_vec1+time_init;
%% a-IRG Algorithm
param.maxtime = time_vec1(end);
param.maxiter = 1e7;

gamma = 5;
eta = .1;
param.gamma = gamma; % the stepsize for the lower-level
param.eta = eta; % the relative stepsize for the upper-level

disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast,tsa_AP,p_AP] = Alg_projection(fun_f,grad_f,grad_g,...
    fun_g,@(x) acc_p_rule(X_test,y_test,x_control_test,x),param,x_init);
disp('a-IRG Solution Achieved!');

%% BiG-SAM Algorithm
param.eta_g = 0.1; % stepsize for upper-level
param.eta_f = 0.1; % stepsize for lower-level
param.gamma = 1;
disp('BiG-SAM Algorithm starts');
[f_vec3,g_vec3,time_vec3,xlast3,tsa_SAM,p_SAM] = BigSAM(fun_f,grad_f,grad_g,fun_g,...
    @(x) acc_p_rule(X_test,y_test,x_control_test,x),param,x_init);
disp('BiG-SAM Solution Achieved!');

%% DBGD Algorithm
param.alpha = 1;
param.beta = 1;
param.stepsize = 8e-2;
param.maxiter=1e7;
disp('DBGD Algorithm starts');
[f_vec4,g_vec4,time_vec4,xlast4,tsa_DBGD,p_DBGD] = DBGD(fun_f,grad_f,grad_g,fun_g,...
    @(x) acc_p_rule(X_test,y_test,x_control_test,x),param,x_init);
disp('DBGD Solution Achieved!');

%% Figures

%% lower-level gap
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

N_marker = 10;
time_idx = linspace(0,param.maxtime,N_marker);
marker_idx1 = zeros(N_marker,1);
marker_idx2 = zeros(N_marker,1);
marker_idx3 = zeros(N_marker,1);
marker_idx4 = zeros(N_marker,1);
for j=1:N_marker
    [~,idx] = min(abs(time_vec1-time_idx(j)));
    marker_idx1(j) = idx;
    [~,idx] = min(abs(time_vec2-time_idx(j)));
    marker_idx2(j) = idx;
    [~,idx] = min(abs(time_vec3-time_idx(j)));
    marker_idx3(j) = idx;
    [~,idx] = min(abs(time_vec4-time_idx(j)));
    marker_idx4(j) = idx;
end

semilogy(time_vec1,(g_vec1-gstar),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
semilogy(time_vec3,(g_vec3-gstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec2,(g_vec2-gstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec4,abs(g_vec4-gstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx4, 'Color',"#77AC30");
ylabel('$g(\beta_k)-g^*$')
xlabel('time (s)')
set(gca,'FontSize',24);
legend('Interpreter','latex','Location','northeast')
grid on;
grid minor
pbaspect([1 0.8 1])

% print('-depsc2','-r600','./figs/nc_lower_time.eps')
%% upper-level objective
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec1,(f_vec1),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
% disregard function values that are too small (in the initial stage of the algorithm)
start_idx3 = find(f_vec3>1e-5,1);
semilogy(time_vec3(start_idx3:end),(f_vec3(start_idx3:end)),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3(marker_idx3>=start_idx3)-start_idx3+1);
start_idx2 = find(f_vec2>1e-5,1);
semilogy(time_vec2(start_idx2:end),(f_vec2(start_idx2:end)),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2(marker_idx2>=start_idx2)-start_idx2+1);
start_idx4 = find(f_vec4>1e-5,1);
semilogy(time_vec4(start_idx4:end),f_vec4(start_idx4:end),'>-','DisplayName','DBGD','MarkerIndices', marker_idx4(marker_idx4>=start_idx4)-start_idx4+1, 'Color',"#77AC30");

ylabel('$f(\beta_k)$')
xlabel('time (s)')
set(gca,'FontSize',24);
legend('Interpreter','latex','Location','southeast')
grid on;
grid minor
pbaspect([1 0.8 1])

% print('-depsc2','-r600','./figs/nc_upper_obj_time.eps')
%% test error
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec1,tsa_BFW,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
start_idx3 = find(tsa_SAM>0.7,1);
semilogy(time_vec3(start_idx3:end),tsa_SAM(start_idx3:end),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3(marker_idx3>=start_idx3)-start_idx3+1);
start_idx2 = find(tsa_AP>0.7,1);
semilogy(time_vec2(start_idx2:end),tsa_AP(start_idx2:end),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2(marker_idx2>=start_idx2)-start_idx2+1);
start_idx4 = find(tsa_DBGD>0.7,1);
semilogy(time_vec4(start_idx4:end),tsa_DBGD(start_idx4:end),'>-','DisplayName','DBGD','MarkerIndices', marker_idx4(marker_idx4>=start_idx4)-start_idx2+1, 'Color',"#77AC30");

ylabel('Accuracy')
xlabel('time (s)')
set(gca,'FontSize',24);
legend('Interpreter','latex','Location','southeast')
grid on;
grid minor
pbaspect([1 0.8 1])
% print('-depsc2','-r600','./figs/nc_test_error_time.eps')

%% p-rule
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

plot(time_vec1,p_BFW,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
start_idx3 = find(p_SAM == 0,1,'last')+1;
plot(time_vec3(start_idx3:end),p_SAM(start_idx3:end),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3(marker_idx3>=start_idx3)-start_idx3+1);
start_idx2 = 3;
plot(time_vec2(start_idx2:end),p_AP(start_idx2:end),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2(marker_idx2>=start_idx2)-start_idx2+1);
start_idx4 = find(p_DBGD == 0,1,'last')+1;
plot(time_vec4(start_idx4:end),p_DBGD(start_idx4:end),'>-','DisplayName','DBGD','MarkerIndices', marker_idx4(marker_idx4>=start_idx4)-start_idx4+1, 'Color',"#77AC30")

ylabel('$p\%$-rule')
xlabel('time (s)')
set(gca,'FontSize',24);
legend('Interpreter','latex','Location','southeast')
grid on;
grid minor
pbaspect([1 0.8 1])

% print('-depsc2','-r600','./figs/nc_p_rule_time.eps')

function out = log_logistic(X)
out = 0*X;
idx = X>0;
out(idx) = -log(1+exp(-X(idx)));
out(~idx) = X(~idx)-log(1+exp(X(~idx)));
end

function out = logistic(X)
out = 0*X;
idx = X>0;
out(idx) = 1./(1+exp(-X(idx)));
out(~idx) = exp(X(~idx))./(1+exp(X(~idx)));
end

% compute the accuracy as well as the p-rule
function [acc,p] = acc_p_rule(X_test,y_test,x_control,x)
test_total = length(y_test);
y_hat = sign(X_test*x);
n_rand = sum(y_hat == 0);
label_rand = 2*randi([0,1],n_rand,1)-1;
y_hat(y_hat == 0) = label_rand;
acc = sum(y_hat == y_test)/test_total;

non_prot_all = sum(x_control == 1);
prot_all = sum(x_control == 0);
non_prot_pos = sum(y_hat(x_control == 1) == 1);
prot_pos = sum(y_hat(x_control == 0) == 1);
frac_non_prot_pos = non_prot_pos/non_prot_all;
frac_prot_pos = prot_pos/prot_all;

if frac_non_prot_pos == 0 && frac_prot_pos == 0
    p = 100;
else
    p = min(frac_non_prot_pos/frac_prot_pos,frac_prot_pos/frac_non_prot_pos)*100;
end

end