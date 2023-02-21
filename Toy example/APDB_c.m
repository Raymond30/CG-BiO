function [x,subopt_vec,infeas_vec,x_hist]=...
    APDB_c(f,grad_f,g,grad_g,x0,y0,g0,stop_criteria,max_iter,fstar)
%************************************************************
% IMPORTANT: optimal solution is required to measure the accuracy of the
% solution found

% Written by Erfan Yazdandoost Hamedani, Penn State University

% created on 14 November 2018.

% If acc=false then the step-sizes are constant according to the rule  
% Theorem 2.1 part (I) in the paper https://arxiv.org/pdf/1803.01401.pdf
% If acc=true then the step-sizes are non-constant according to the rule  
% Theorem 2.1 part (II) in the paper.
%************************************************************
% min_{x>=0, <b,x>=0} max_{y in simplex} L(x,y)
% where L(x,y)=-2<x,ones> + sum_{l=1}^M (c/r_l) y(l)*x^T*G(K_l^{tr})*x +
% lambda*norm(x)^2
%************************************************************
% stop_type: 1 - Number of iterations (max_iter)
%            2 - norm of consecutive iterates (stop_criteria)
%************************************************************
    %---------- Unfolding Input ----------%
    sc = 0;
    n = length(x0);
    m = length(y0);
    %-------------------------------------%
    disp('**********************************************************')
    disp(['APDB Convex'])
    disp('**********************************************************') 
    %------------ step-size Parmeters------%
    eta = 0.7;
    tau = 1e-2;
    gamma = 1;
    sigma_old = gamma*tau;
    tau_old = tau;
    mu = sc;
    %------ Initialization ----------------%
    subopt_vec = [];
    infeas_vec = [];
    x_hist = x0';
    y = y0;
    x = x0;
    iter = 0;
    grad_y_full = @(x) [(g(x)-g0);[1,1;4,6]*x-[1;5]];
    grad_x_full = @(z) grad_f(z(1:n))+grad_g(z(1:n))*z(n+1)+[1;1]*z(n+2)+...
        [4;6]*z(n+3);
    G_y = grad_y_full(x);
    tic;
    %---------- Main Algorithm ------------%    
    while iter<=max_iter
        iter = iter+1;
        G_y_old = G_y;
        G_y = grad_y_full(x);
        %orc = orc+1;
        while true
            sigma = gamma*tau;
            theta = sigma_old/sigma;
            ytild = max(0,y+sigma*((1+theta)*G_y-theta*G_y_old));
            G_x = grad_x_full([x;ytild]);
            xtild = max(0,x-tau*G_x);
            xdiff = xtild-x;
            E = (grad_x_full([xtild;ytild])-G_x)'*xdiff+...
                (norm(grad_y_full(xtild)-G_y)^2)*sigma*2;
            E = E-norm(xdiff)^2/(2*tau);
            if E<=0
                break;
            else
                %inner(iter)=inner(iter)+1;
                tau = tau*eta;
            end
        end
        gamma_old = gamma;
        gamma = gamma*(1+mu*tau);
        %tau = tau*sqrt(gamma_old/gamma *(1+theta));
        tau_new = tau*sqrt(gamma_old/gamma *(1+tau/tau_old));
        sigma_old = sigma;
        tau_old = tau;
        tau = tau_new;
        
        y = ytild;
        x = xtild;

        x_hist = [x_hist;x'];
        subopt = (f(x));
        %subopt = obj_fun(x,y,A,b,rho);
        subopt_vec(iter,1) = subopt;
        infeas = g(x);%sum(pos(grad_y_full(x)));
        infeas_vec(iter,1) = infeas;
        
        if subopt-fstar<=stop_criteria && sum(pos(grad_y_full(x)))<=stop_criteria
            break;
        end
    end
%     fprintf(...
%         'Iteration    Time    f(x)\n');
%     fprintf('%d    %9.4f       %9.1e\n',iter,time_period(epoch_counter),...
%         subopt_vec(epoch_counter));
%     figure(1);
%     semilogy(iter_epoch,subopt_vec,'-');
%     figure(2);
%     semilogy(time_period,subopt_vec,'-');
end