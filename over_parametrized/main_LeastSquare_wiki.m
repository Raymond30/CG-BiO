clear;
%% load data
seed = 123456;

rng(seed);
load('wikivital_mathematics.mat');
[m,n]= size(A);

% training model data
trp=0.6; % 60% of data is used as the training set
%          % 20% of data is used as the validation set
%          % 20% of data is used as the test set
% 
idx=randperm(m); %for randomly scatter the dataset
A2=A(idx(1:round(trp*m)),:);
b2=b(idx(1:round(trp*m)),:);
% validation model data 
A1=A(idx(round(trp*m)+1:(round(trp*m+(m*(1-trp)/2)))),:);
b1=b(idx(round(trp*m)+1:(round(trp*m+(m*(1-trp)/2)))),:);

% test model data
A3=A(idx(round((trp*m)+(m*(1-trp)/2))+1:end),:);
b3=b(idx(round((trp*m)+(m*(1-trp)/2))+1:end),:);

epsilon_f = 1e-4;
epsilon_g = 1e-4;
% global lambda;
lambda=1;
x_init = sparse(n,1);
maxiter=1e5;
%% function definition
fun_f= @(x) sum_square(A1*x-b1)/2;
fun_g = @(x) sum_square(A2*x-b2)/2;
grad_g = @(x) A2'*(A2*x-b2);
grad_f= @(x) A1'*(A1*x-b1);
 
% Finding optimal solution
cvx_begin quiet
variables xstar_g(n,1)
cvx_precision high
minimize fun_g(xstar_g)
subject to
    norm(xstar_g,1)<=lambda;
cvx_end
gstar = cvx_optval;

cvx_begin quiet
variables xstar(n,1)
cvx_precision high
cvx_solver sdpt3
minimize fun_f(xstar)
subject to
    norm(xstar,1)<=lambda;
    fun_g(xstar)<=gstar;
cvx_end
cvx_optval
fstar = cvx_optval;
%% CG for the sub problem
param.epsilong = epsilon_g/2;
param.lam1=lambda;
param.maxiter=1e4;
tic;
[last_iter , f_hist] = CG_lowerlevel(fun_g,grad_g,x_init,param);
time_init = toc;

%% CG-BiO algorithm
param.epsilonf = epsilon_f;
param.epsilong = epsilon_g;
param.lam=lambda;
param.fun_g_x0 = fun_g(last_iter);
param.maxiter=1e4;

disp('CG-BiO starts');
[f_vec1,g_vec1,time_vec1,x,tsa_BCG] = CG_BiO(fun_f,grad_f,grad_g,fun_g,...
    @(x)TSA_LS(x,A3,b3),param,last_iter);
disp('CG-BiO Achieved!');

time_vec1 = time_vec1+time_init;
%% a-IRG Algorithm
param.maxtime = time_vec1(end);
param.maxiter=1e7;

disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast,tsa_AP] = Alg_Projection(fun_f,grad_f,grad_g,...
    fun_g,@(x)TSA_LS(x,A3,b3),param,x_init);
disp('a-IRG Solution Achieved!');

%% BiG-SAM Algorithm
param.eta_g = 1/eigs(A2'*A2,1);
param.eta_f = 2/eigs(A1'*A1,1);
param.gamma = 10;
disp('BiG-SAM Algorithm starts');
[f_vec3,g_vec3,time_vec3,xlast3,tsa_SAM] = BigSAM(fun_f,grad_f,grad_g,fun_g,@(x)TSA_LS(x,A3,b3),param,x_init);
disp('BiG-SAM Solution Achieved!');

%% DBGD
param.alpha = 1;
param.beta = 1;
param.stepsize = 1e-4;
param.maxiter=1e7;
param.maxtime = time_vec1(end);
disp('DBGD Algorithm starts');
[f_vec5,g_vec5,time_vec5,xlast5,tsa_DBGD] = DBGD(fun_f,grad_f,grad_g,fun_g,@(x)TSA_LS(x,A3,b3),param,x_init);
disp('DBGD Solution Achieved!');

%% MNG
param.maxtime = time_vec1(end);
param.maxiter=length(time_vec1);
param.M = eigs(A2'*A2,1);

[f_vec4,g_vec4,time_vec4,xlast4,tsa_MNG] = MNG(A1,b1,fun_g,grad_g,@(x)TSA_LS(x,A3,b3),param,A1\b1);



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
marker_idx5 = zeros(N_marker,1);
marker_idx_GD = zeros(N_marker,1);
for j=1:N_marker
    [~,idx] = min(abs(time_vec1-time_idx(j)));
    marker_idx1(j) = idx;
    [~,idx] = min(abs(time_vec2-time_idx(j)));
    marker_idx2(j) = idx;
    [~,idx] = min(abs(time_vec3-time_idx(j)));
    marker_idx3(j) = idx;
    [~,idx] = min(abs(time_vec4-time_idx(j)));
   marker_idx4(j) = idx;
   [~,idx] = min(abs(time_vec5-time_idx(j)));
   marker_idx5(j) = idx;
end

semilogy(time_vec1,abs(g_vec1-gstar),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
semilogy(time_vec3,abs(g_vec3-gstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec2,abs(g_vec2-gstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec4,abs(g_vec4-gstar),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,abs(g_vec5-gstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5, 'Color',"#77AC30");

ylabel('$|g(\beta_k)-g^*|$')
xlabel('time (s)')
set(gca,'FontSize',24);
set(gca,'YLim',[1e-6,10])
legend('Interpreter','latex','Location','southwest')
grid on;
grid minor
pbaspect([1 0.8 1])

% print('-depsc2','-r600','./figs/lower_subopt_time.eps')
%% upper-level gap
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec1,abs(f_vec1-fstar),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
semilogy(time_vec3,abs(f_vec3-fstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec2,abs(f_vec2-fstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec4,abs(f_vec4-fstar),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,abs(f_vec5-fstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5, 'Color',"#77AC30");

ylabel('$|f(\beta_k)-f^*|$')
xlabel('time (s)')
% set(gca,'YLim',[1e-11,1e3])
set(gca,'FontSize',24);
% set(gca,'YLim',[0,1])
legend('Interpreter','latex','Location','northeast')
grid on;
grid minor
pbaspect([1 0.8 1])

% print('-depsc2','-r600','./figs/upper_subopt_time.eps')
%% test error
figure;
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',5)
set(gcf,'DefaultLineMarkerSize',16);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec1,tsa_BCG,'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
hold on;
semilogy(time_vec3,tsa_SAM,'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec2,tsa_AP,'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec4,tsa_MNG,'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,tsa_DBGD,'>-','DisplayName','DBGD','MarkerIndices', marker_idx5, 'Color',"#77AC30");


ylabel('Test error')
xlabel('time (s)')
set(gca,'FontSize',24);
% set(gca,'YLim',[0,1])
legend('Interpreter','latex','Location','northeast')
grid on;
grid minor
pbaspect([1 0.8 1])
% print('-depsc2','-r600','./figs/test_error_time.eps')
