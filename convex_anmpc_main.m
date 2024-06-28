%% Set up the random number generator
if ~exist('stream','var')
  % Either get the current global stream's state:
  %   stream = RandStream.getGlobalStream;
  % Or reset it using a seed based on current time:
  stream = RandStream('mt19937ar','Seed','shuffle');
end
RandStream.setGlobalStream(stream);
% If random number generator state was previously saved
% check whether new data is required
if exist('savedState','var')
  newdata = input('New data? [y/n]: ','s');
end
% Either allow new data and save state
% Or reset seed back to saved state
if exist('newdata','var') && ~isempty(newdata) && newdata(1) == 'n'
  stream.State = savedState;
else
  savedState = stream.State;
end

%% Simulation parameters
p = param_init();                  % Initialise problem parameters 
T_sim = 20;                         % Number of time steps simulated
Tmax = 50;                         % Number of time steps in MPC horizon
p.delta = 1;                       % discrete time sampling interval
maxiter = 5;                       % Max number of iterations
maxiterLS = 10;                    % Max number of line search iterations
alphaLS = 0.2;                      % Line search scaling factor
Q = [0 0;0 1]; R = 0.1;            % Cost matrices
sqrtQ = [0 1]; sqrtR = R^0.5; 
dtheta = 1; p.dtheta = dtheta;
theta_mu = 0; %1e-5;               % Parameter update gain: theta_mu = 0 => no parameter adaptation

%% Initialise problem
x_sim = zeros(p.nx, T_sim+1);        % closed loop state sequence
u_sim = zeros(p.nu, T_sim);        % closed loop control sequence
Iter_sim = zeros(1,T_sim);
J_sim = zeros(1,T_sim);

x_0 = zeros(p.nx, Tmax+1);         % nomimal state sequence
u_0 = zeros(p.nu, Tmax);           % nomimal control sequence
x_r = p.h_r.*ones(p.nx, T_sim+Tmax);     % reference state
u_r = p.u_r.*ones(p.nu, T_sim+Tmax);     % reference input
t_avg= 0;                          % counter for average time / iteration
iter_count = 0;                    % counter for number of iterations

%% DC decomposition
% Define RBF 
N_samples = 1000; 
sqt_N_RBF = 5;
N_RBF = sqt_N_RBF^2;               % number of RBF (only works in 2D)
[X1_RBF, X2_RBF] = meshgrid(linspace(p.x_min(1)-3, p.x_max(1)+3, sqt_N_RBF),...
                            linspace(p.x_min(1)-3, p.x_max(1)+3, sqt_N_RBF));
c_RBF = [X1_RBF(:)';X2_RBF(:)'];   % RBF centers
rho_RBF = ones(N_RBF);             % RBF scalings 

% Generate training data
[x_train, u_train, y_train] = gen_train(@dynamics, N_samples, p);
input_train = {[x_train(1, :); u_train], x_train}; 
% input to the RBF for each state (needs adaptation to specific dynamics)
  
% Fit RBF and get decomposition
[f_RBF_, g_RBF, h_RBF, theta,... 
 theta_g, theta_h, c_g, c_h,...
 rho_g, rho_h, MAE_train] = get_RBF(N_samples, c_RBF, rho_RBF, ...
                                    p, input_train, y_train); 

% Function wrappers (problem specific)
f = @(x, u, p) ([f_RBF_{1}([x(1, :); u]); f_RBF_{2}(x)]);
g_ = @(x, u, p) ([g_RBF{1}([x(1, :); u]); g_RBF{2}(x)]);
h_ = @(x, u, p) ([h_RBF{1}([x(1, :); u]); h_RBF{2}(x)]);

% Test fit (will only work for specific coupled tank problem)
plt = false; 
dim_N_test = 10; % test points per dimension
p_test_fit = p;
p_test_fit.x_max = p.x_max - [0;0];
p_test_fit.x_min = p.x_min + [10;10];
MAE = test_fit(@dynamics, dim_N_test, f_RBF_, g_RBF, h_RBF, p_test_fit, plt); 
p.w_max = 1*MAE;
% *** change p.w_max to introduce artificial disturbance ***

% theta_g = theta_g_old;
% theta_h = theta_h_old;
theta = [theta_g{1}; theta_g{2}; theta_h{1}; theta_h{2}];
theta_store = repmat(theta,1,T_sim+1);

%% Terminal set
% Linearise g at reference
[A1_term, B1_term] = linearise(p.h_r, p.u_r, theta_g, c_g, rho_g, p);

 % Linearise h at reference 
[A2_term, B2_term] = linearise(p.h_r, p.u_r, theta_h, c_h, rho_h, p); 

% Discretise
A_d_term = eye(p.nx) + p.delta*(A1_term - A2_term);
B_d_term = p.delta*(B1_term - B2_term);

% Set optimisation
[K,P,V,gam,beta_cost] = term_comp(A_d_term, B_d_term,Q,R,p);
sqrtP = sqrtm(P); sqrtV = sqrtm(V); 

%p.A1 = p.A1*0.8; % reduce flow between tanks by 20%

%% Feasible inital trajectory
x_0(:,1) = p.x_init; % initial state
% Q_init = Q; R_init = 0.2;
% [K_init,P_init] = dlqr(A_d_term,B_d_term,Q_init,R_init); K_init = -K_init;
K_init = K; Q_dp = diag([1e3,1]); R_dp = 1e-3;
K_dp_tol = 1e-3;
iterflag = 1; iter = 0; dp_itermax = 5;
while iterflag
    if iter == 0
        K_dp = repmat(K_init,1,1,Tmax);
    else
        [A1, B1] = linearise(x_0, u_0, theta_g, c_g, rho_g, p); % linearise g
        [A2, B2] = linearise(x_0, u_0, theta_h, c_h, rho_h, p); % linearise h
        K_dp_old = K_dp(:,:,1);
        A_d = repmat(eye(p.nx),1,1,Tmax) + p.delta*(A1 - A2);
        B_d = p.delta*(B1 - B2);
        K_dp = dp_seq(A_d,B_d,Q_dp,R_dp,P);
    end
    Jsq_init = 0;
    for i = 1:Tmax
        % Control input (dummy controller)
        u_0(:,i) = max(min(K_dp(:,:,i)*(x_0(:,i)-x_r(:,i)) + u_r(:,i), p.u_max), p.u_min);
    
        % Generate feasible trajectory
        % [~, x_0(:,i+1)] = dynamics(x_0(:,i), u_0(:,i), p);
        x_0(:,i+1) = x_0(:,i) + p.delta*(f_RBF(x_0(:,i), u_0(:,i), theta_g, c_g, rho_g) ...
                       - f_RBF(x_0(:,i), u_0(:,i), theta_h, c_h, rho_h));
        Jsq_init = Jsq_init + (norm(sqrtQ*(x_0(:,i)-x_r(:,i)),2))^2 + (norm(sqrtQ*(u_0(:,i)-u_r(:,i)),2))^2;
    end 
    Jsq_init = Jsq_init + (norm(sqrtP*(x_0(:,i)-x_r(:,i)),2))^2;
    iter = iter+1;
    if iter > dp_itermax || (iter > 1 && norm(K_dp(:,:,1) - K_dp_old)/norm(K_dp_old) < K_dp_tol)
        iterflag = 0;
    else
        plot_init(x_0,u_0,x_r,V,p);
        if iter >  1
            fprintf("iter: %d/%d J_init = %.3e norm(K_dp - K_dp_old)/norm(K_dp_old) = %.3e\n", iter, dp_itermax, sqrt(Jsq_init), norm(K_dp(:,:,1) - K_dp_old)/norm(K_dp_old));
        end
        u_00 = u_0;
    end
end

% initial seed for the nominal input perturbation
K_dp_c = num2cell(K_dp,[1,2]); 
K_dp_ = blkdiag(K_dp_c{:});
c_0 = u_0 - reshape(K_dp_ * reshape(x_0(:,1:Tmax), [p.nx*Tmax, 1]), [p.nu, Tmax]);
c_00 = c_0;
x0_00 = x_0(:,1);
delta_c_init = 1e-6; % placeholder
delta_x_init = 1e-6; % placeholder
Jold = sqrt(Jsq_init);
K_dp_00 = K_dp;
plot_init(x_0,u_0,x_r,V,p);

[c, x_lb, x_ub, Termconstr, cvx_status] = cvx_terminal_optimisation(x_0, u_0, c_0, x_r(:,1:Tmax+1), A1, A2, B1, B2, K_dp, sqrtV, p, theta_g, c_g, rho_g, theta_h, c_h, rho_h);
if Termconstr > 1
    figure(1); 
    plot(x_lb(1,:),x_lb(2,:),'m+');
    plot(x_ub(1,:),x_ub(2,:),'c+');
    error("Failed to find feasible solution: minimum terminal constraint x_term'*V*x_term = %.3e", Termconstr); % give up
end

%% MPC loop
ctol = 1e-2;
Jtol = 1e-2;
t = 0; 
x_sim(:,1) = p.x_init; % initial state
while t < T_sim
    fprintf("************** Solving problem at time %d/%d ******************\n", t, T_sim)

    iter = 1; iterLS = 1; iterflag = 1;
    while iterflag
        fprintf("iteration %d/%d (t=%d) ", iter, maxiter, t)

        % Compute nominal trajectory and linearisation, horizon: {t+1:t+Tmax, t+Tmax+1}
        if iter == 1 && iterLS == 1 && t > 0
            c_term = u_r(:,t+Tmax) - K*x_r(:,t+Tmax);
            c_0 = [c_0(2:Tmax), c_term];
            c_00 = [c_00(2:Tmax), c_term];
            x0_00 = x_0(:,2);
            x_0(:,1) = x_sim(:,t+1);
            % delta_c_init = norm(c_0 - c_00);
            % delta_x_init = norm(x_0(:,1) - x0_00);
            Jold = sqrt(Jold^2 - (norm([sqrtQ*(x_sim(:,t)-x_r(:,t));sqrtR*(u_sim(:,t)-u_r(:,t))]))^2 + beta_cost);
            K_dp(:,:,1:end-1) = K_dp(:,:,2:end);
            K_dp(:,:,end) = K;
        end

        for i = 1:Tmax
            u_0(:,i) = K_dp(:,:,i)*x_0(:,i) + c_0(:,i);
            x_0(:,i+1) = x_0(:,i) + p.delta*(f_RBF(x_0(:,i), u_0(:,i), theta_g, c_g, rho_g) ...
                - f_RBF(x_0(:,i), u_0(:,i), theta_h, c_h, rho_h));
            % [~, x_0(:,i+1)] = dynamics(x_0(:,i), u_0(:,i), p);
        end

        % Linearise system (CT)
        [A1, B1] = linearise(x_0, u_0, theta_g, c_g, rho_g, p); % linearise g
        [A2, B2] = linearise(x_0, u_0, theta_h, c_h, rho_h, p); % linearise h
        A_d = repmat(eye(p.nx),1,1,Tmax) + p.delta*(A1 - A2);
        B_d = p.delta*(B1 - B2);
        K_dp = dp_seq(A_d,B_d,Q_dp,R_dp,P);

        % N = Tmax - t;
        % if N > 0
        %     A_d = repmat(eye(p.nx),1,1,N) + p.delta*(A1(:,:,1:N) - A2(:,:,1:N));
        %     B_d = p.delta*(B1(:,:,1:N) - B2(:,:,1:N));
        %     K_dp(:,:,1:N) = dp_seq(A_d,B_d,Q,R,P);
        % end
        % N = max(N,0);
        % K_dp(:,:,N+1:Tmax) = repmat(K, 1, 1, Tmax-N);

        % Closed loop
        % Phi1 = A1 + B1.*K;
        % Phi2 = A2 + B2.*K;

        % Optimisation
        tic
        [c, x_lb, x_ub, J, cvx_status] = cvx_optimisation(x_0, u_0, c_0, ...
            x_r(:,t+(1:Tmax+1)), u_r(:,t+(1:Tmax)), A1, A2, B1, B2, K_dp, ...
            sqrtR, sqrtQ, sqrtV, sqrtP, p, ...
            theta_g, c_g, rho_g, theta_h, c_h, rho_h, 1.05*Jold);

        % [c, x_lb, x_ub, info] = mpc_ocp(x_0, u_0, c_0, ...
        %     x_r, u_r, Phi1, Phi2, B1, B2, K, ...
        %     sqrtR, sqrtQ, sqrtV, sqrtP, p, ...
        %     theta_g, c_g, rho_g, theta_h, c_h, rho_h);

        t_elapsed = toc;
        fprintf("cputime %.3e %s ", t_elapsed, cvx_status);
        t_avg = t_avg + t_elapsed;
        iter_count = iter_count+1;

        % Update initial linearisation state and control input perturbation
        if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
            c_0 = c_00 + alphaLS*(c_0 - c_00);
            x_0(:,1) = x0_00 + alphaLS*(x_0(:,1) - x0_00);
            K_dp = K_dp_00;
            delta_c = norm(c_0 - c_00)/norm(c_00);
            delta_x0 = norm(x_0(:,1) - x0_00);
            fprintf("| iterLS %d/%d ", iterLS, maxiterLS);
            iterLS = iterLS + 1;
            if iterLS > maxiterLS || (t == 0 && iter == 1)
                fprintf("\n");
                error("Failed to find feasible solution"); % give up
            end
        else
            deltaJ = abs(J - Jold)/Jold;
            % Jold = J; % only update the cost bound at the first iteration
            % of each timestep
            c_00 = c_0;
            x0_00 = x_0(:,1);
            K_dp_00 = K_dp;
            c_0 = c_0 + c;
            delta_c = norm(c_0 - c_00)/norm(c_00);
            delta_x0 = norm(x_0(:,1) - x0_00);
            iter = iter + 1;
            if (deltaJ < Jtol && delta_c < ctol) || iter > maxiter
                iterflag = 0;
            end
        end
        fprintf(1,"delta_c %.3e delta_x0 %.3e deltaJ %.3e J %.3e\n", delta_c, delta_x0, deltaJ, Jold);

    end

    Iter_sim(t+1) = iter;
    J_sim(t+1) = J;

    % Update control input and system state
    u_sim(t+1) = K_dp(:,:,1)*x_sim(:,t+1) + c_0(1); % Control input
    % x_next = x_sim(:,t+1) + p.delta*(f_RBF(x_sim(:,t+1), u_sim(:,t+1), theta_g, c_g, rho_g) ...
    %               - f_RBF(x_sim(:,t+1), u_sim(:,t+1), theta_h, c_h, rho_h));
    [~, x_next] = dynamics(x_sim(:,t+1), u_sim(t+1), p); % State
    x_sim(:,t+2) = x_next + (p.w_max).*randn(p.nx,1);

    % parameter update using Least Mean Square (LMS) filter
    if theta_mu > 0
        [g_, D_g_] = f_RBF(x_sim(:,t+1), u_sim(:,t+1), theta_g, c_g, rho_g);
        [h_, D_h_] = f_RBF(x_sim(:,t+1), u_sim(:,t+1), theta_h, c_h, rho_h);
        D_pred = p.delta*[D_g_, -D_h_];
        theta = [theta_g{1}; theta_g{2}; theta_h{1}; theta_h{2}];
        x_next_pred = x_sim(:,t+1) + D_pred*theta; %= x_sim(:,t+1) + p.delta*(g_ - h_);
        normsq_D = norm(D_pred,2)^2;
        while theta_mu > 0.1/normsq_D
            theta_mu = 0.1*theta_mu;    % reduce parameter update gain
        end
        delta_theta = theta_mu*D_pred'*(x_sim(:,t+2) - x_next_pred); % LMS parameter update
        theta = max(theta + delta_theta,0);
        ntheta = 0;
        theta_g{1} = theta(ntheta+(1:length(theta_g{1})));
        ntheta = ntheta + length(theta_g{1});
        theta_g{2} = theta(ntheta+(1:length(theta_g{2})));
        ntheta = ntheta + length(theta_g{2});
        theta_h{1} = theta(ntheta+(1:length(theta_h{1})));
        ntheta = ntheta + length(theta_h{1});
        theta_h{2} = theta(ntheta+(1:length(theta_h{2})));
        theta_store(:,t+2) = theta;
        fprintf(1,"delta_theta %.3e 1/norm(D)^2 %.3e theta_mu %.3e\n", norm(delta_theta)/norm(theta), 1/normsq_D, theta_mu);
    end

    t = t+1;
    fprintf(1,"\n");
    % *** 
    % to include an extra disturbance change this to (e.g.)     
    % w_true = 2*(rand(2,1)-0.5).*p.w_max;
    % x_sim(:,t+1) = x_next + w_true;
    % ***

    % x(:,t+1) = x(:,t) + p.delta*(f_RBF(x(:,t), u_opt(t), theta_g, ...
    % c_g, rho_g) - f_RBF(x(:,t), u_opt(t), theta_h, c_h, rho_h) + w_true);

end

J_run = norm([sqrtQ*(x_sim-x_r(:,1:t+1)),sqrtR*(u_sim-u_r(:,t))]);
fprintf('Average time per iteration of the optimisation: %.2f s\n', t_avg/iter_count);
fprintf('Closed loop cost: %.3e\n', J_run);

%% Plot results
t = (0:T_sim)*p.delta;
figure
subplot(3,1,1)
stairs(t, [u_sim u_sim(end)],'b','LineWidth',1)
hold on
plot(t,[p.u_min;p.u_max]*ones(1,T_sim+1),'--')
%axis([0 t(end) min(u_sim)/1.2 max(u_sim)*1.2])
ylabel('Control $u_t$ (V)', 'Interpreter','latex')
grid on
subplot(3,1,2)
plot(t,x_sim(1,:),'b','LineWidth',1)
hold on
plot(t,[p.x_min(1);p.x_max(1)]*ones(1,T_sim+1),'--')
%axis([0 t(end) 0 20])
ylabel('State $[x_t]_1$ (cm)', 'Interpreter','latex')
grid on
subplot(3,1,3)
plot(t,x_sim(2,:),'b','LineWidth',1)
hold on
plot(t,[p.x_min(2);p.x_max(2)]*ones(1,T_sim+1),'--')
plot(t,p.h_r(2)*ones(1,T_sim+1),'-.')
%axis([0 t(end) 0 20])
ylabel('State $[x_t]_2$ (cm)', 'Interpreter','latex')
xlabel('time $t$ (s)', 'Interpreter','latex')
grid on
%figure(1); hold on;
%plot(x_sim(1,:),x_sim(2,:));