function [u_0,x_0] = update_nominal_trajectory(c_0,u_0,x_0,K,Tmax,p)

x_ = x_0(:, 1);
for i=1:Tmax
    u_0(:,i) = c_0(:,i) + K*x_; 
    x_0(:,i+1) = x_ + p.delta*(f_RBF(x_, u_0(:,i), theta_g, c_g, rho_g) ...
                 - f_RBF(x_, u_0(:,i), theta_h, c_h, rho_h));
    x_ = x_0(:,i+1);
end 
