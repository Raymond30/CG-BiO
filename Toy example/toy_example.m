clear;
clc;
seed = 12345;
rng(seed);

n = 2;
m = 2;

% Toy example:
% minimize 0.5x1^2-0.5x1+0.1x2
% s.t. x in argmin{-(x1+x2) | x1+x2<=1, 4*x1+6*x2<=5, x1>=0, x2>=0}
% The lower-level solution set is X*_g={(x1,x2) | x1+x2=1, x1 in [0.5,1], x2 in [0,0.5]}
% The optimal solution is (x1*,x2*)=(0.6,0.4)

%% functions
f = @(x) 0.5*x(1)^2-0.5*x(1)+0.1*x(2);
grad_f = @(x) [x(1)-0.5;0.1];
g = @(x) -(x(1)+x(2));
grad_g = @(x) -ones(n,1);
options = optimoptions('quadprog','Display','none');
proj = @(y) quadprog(eye(n)/2,-y,[1,1;4,6],[1,5],[],[],zeros(n,1),[],y,options);
%% solving lower level
gstar = -1;
fstar = 0.5*0.6^2-0.5*0.6+0.1*0.4;
accuracy = 1e-5/2;
x0 = zeros(n,1);
[last_iter_g,f_hist_grad,x1_hist] = gradient_method2(proj,g,gstar,grad_g,x0,@backtracking,accuracy);
param.epsilong = accuracy;
[last_iter , f_hist,x2_hist] = CG_lowerlevel(g,grad_g,x0,param);
%% solving bilevel
param.epsilonf = accuracy*2;
param.epsilong = accuracy*2;
[f_vec1,g_vec1,x3_hist] = CG_BiO_LR(f,grad_f,grad_g,g,param,last_iter);

[x,subopt_vec,infeas_vec,x4_hist]=...
    APDB_c(f,grad_f,g,grad_g,last_iter_g,zeros(3,1),f_hist_grad(end),param.epsilonf,1e4,fstar);

norm(x3_hist(end,:)-[0.6,0.4])
norm(x4_hist(end,:)-[0.6,0.4])
%% plot
hold on; grid on;
plot([x4_hist(:,1)],[x4_hist(:,2)],'x');
plot([x3_hist(:,1)],[x3_hist(:,2)],'*');
plot([x4_hist(end,1)],[x4_hist(end,2)],'cx');
plot([x3_hist(end,1)],[x3_hist(end,2)],'g*');

figure;
semilogy(abs(f_vec1-f([0.6,0.4])));
hold on; grid on;
semilogy(abs(subopt_vec-f([0.6,0.4])));

figure;
semilogy(abs(g_vec1+1));
hold on; grid on;
semilogy(abs(infeas_vec+1));