function [f_,x_] = dynamics(x, u, param)
%DYNAMICS compute the dynamic update x_ = f(x, u)
%   Input:  - x: state at time k
%           - u: input at time k
%           - param: structure of parameters
%   Output: - f_: dx/dt at time k+1
%             x_: state at time k+1

f1 = (param.k_p*u - param.A1*sqrt(2*param.g*x(1, :)))/param.A;
f2 = (param.A1*sqrt(2*param.g*x(1, :)) - param.A2*sqrt(2*param.g*x(2, :)))/param.A;

f_ = [f1; f2]; 
x_ = x + param.delta*f_;

