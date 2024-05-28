function [c, xlb, xub, cvx_optval, cvx_status] = cvx_terminal_optimisation(x_0, u_0, c_0, x_r, A1, A2, B1, B2, K, sqrtV, p, theta_g, c_g, rho_g, theta_h, c_h, rho_h)
%CVX_Terminal_OPTIMISATION minimise terminal constraint violation

% Initialisation
[nu, N] = size(u_0);    % number of inputs / horizon 
nx = size(x_0, 1);      % number of states       
nv = 2^nx;              % number of vertices
x_r_term = x_r(:,end);
x_0 = x_0(:, 1:end-1);

% Constraint sets
U_min = p.u_min*ones(1,N); 
U_max = p.u_max*ones(1,N);
X_min = p.x_min*ones(1,N);
X_max = p.x_max*ones(1,N);

% Some useful matrices
A1_c =num2cell(A1,[1,2]); 
A1_ = blkdiag(A1_c{:});
A2_c =num2cell(A2,[1,2]); 
A2_ = blkdiag(A2_c{:});
B1_c =num2cell(B1,[1,2]);
B1_ = blkdiag(B1_c{:});
B2_c =num2cell(B2,[1,2]);
B2_ = blkdiag(B2_c{:});
K_c = num2cell(K,[1,2]);
K_ = blkdiag(K_c{:});

cvx_begin quiet
  cvx_solver mosek
   variables l_n(1) xub(nx, N+1) xlb(nx, N+1) c(nu, N)
   expressions x_vertex(nx, N+1, nv) x(nx, N+1) u(nu, N);
   % minimize(sum(l_x + l_u) + l_n)
   minimize(l_n)
   subject to
   
   % Define vertices
   x_vertex(:, :, 1) = [xlb(1, :); xlb(2, :)];
   x_vertex(:, :, 2) = [xub(1, :); xlb(2, :)];
   x_vertex(:, :, 3) = [xlb(1, :); xub(2, :)];
   x_vertex(:, :, 4) = [xub(1, :); xub(2, :)];

   for l=1:nv
       % Current vertex
       x = x_vertex(:, 1:end-1, l);
       x_term = x_vertex(:, end, l);
       x_ = reshape(x, [nx*N, 1]);
       dx_ = reshape(x - x_0, [nx*N, 1]);

       % Control sequence
       u = reshape(K_ * x_, [nu, N]) + c + c_0;
       du_ = reshape(u - u_0, [nu*N, 1]);

       % Useful variables
       A1_dx = reshape(A1_ * dx_, [nx, N]);
       A2_dx = reshape(A2_ * dx_, [nx, N]);
       B1_du = reshape(B1_ * du_, [nx, N]);
       B2_du = reshape(B2_ * du_, [nx, N]);

       % Objective
       norm(sqrtV*(x_term - x_r_term),2) <= l_n;

       % Input constraints
       U_min <= u;
       U_max >= u;

       % State constraints
       X_min <= x;
       X_max >= x;

       % Initial conditions
       x(:, 1) == x_0(:, 1);

       % Tube constraint
       xlb(:, 2:end) <= x + p.delta*(f_RBF(x_0, u_0, theta_g, c_g, rho_g, 1/p.dtheta) + A1_dx + B1_du ...
           -f_RBF(x, u, theta_h, c_h, rho_h, p.dtheta) - p.w_max.*ones(nx, N)) ;
       xub(:, 2:end) >= x + p.delta*(f_RBF(x, u, theta_g, c_g, rho_g, p.dtheta) ...
           - (f_RBF(x_0, u_0, theta_h, c_h, rho_h, 1/p.dtheta) + A2_dx + B2_du) + p.w_max.*ones(nx, N)) ;

   end

cvx_end
end