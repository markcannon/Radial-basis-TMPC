function [K,P,V, gam, beta_cost] = term_comp(A,B,Q,R,p)

A_term = A; B_term = B; pp.Q = Q; pp.R = R; C = Q(end, :);

n = size(A_term,2);
m = size(B_term,2);

w = zeros(n,4);
w(1:2,1) = [p.w_max(1);p.w_max(2)];
w(1:2,2) = [-p.w_max(1);-p.w_max(2)];
w(1:2,3) = [p.w_max(1);-p.w_max(2)];
w(1:2,4) = [-p.w_max(1);p.w_max(2)];

% Maximize invariant feasible ellipsoid (by minimizing maximum eigenvalue)
% subject to bound on worst case mode 2 cost
lambda_scaling = 4; % 4;

cvx_begin sdp
cvx_solver mosek
variable S(n, n) symmetric
variables Y(m,n) beta_cost(1) gam(1) lambda(1)
minimize(beta_cost - lambda*lambda_scaling)
% minimize(beta_cost)
subject to

S >= lambda*eye(n);
block3 = blkdiag(S,beta_cost);
block4 = blkdiag(S,eye(size(C,1)),inv(pp.R));

for j = 1:4
    block1=[ ((A_term)*S+(B_term)*Y)', S*C' , Y';
              w(:,j)', zeros(1,m+size(C,1))];
    %block2 = [S, Y';zeros(1,m+n)];

    [block3,block1;
        block1',block4] >= 0;
end
cvx_end

Qinv = S;
Q_N=inv(S);
K=Y*Q_N;

% cvx_begin sdp
%   cvx_solver mosek
%   variable gam(1)
%   minimize(-gam)
%   subject to 
%   for q = 1:2
%     F_ = (F(q,:)+G(q)*K);
% 
%     gam <= 1/(F_*Qinv*F_');
%   end
%   for i = 1:2
%     gam <= p.x_term^2/Qinv(i,i);
%   end
%   
%   gam*(Q+K'*R*K) >= double(beta_cost)*Q_N;
% 
% cvx_end


gam = inf;
G=[1/p.u_term; -1/p.u_term]; F=zeros(2,2); h=[1;1];
for q = 1:2
  F_ = (F(q,:)+G(q)*K);
  gam = min(gam,h(q)^2/(F_*Qinv*F_'));
end
G=[1/(p.u_max - p.u_r); 1/(p.u_r - p.u_min)]; F=zeros(2,2); h=[1;1];
for q = 1:2
  F_ = (F(q,:)+G(q)*K);
  gam = min(gam,h(q)^2/(F_*Qinv*F_'));
end
for i = 1:2
  gam = min(gam, p.x_term^2/Qinv(i,i));
end
for i = 1:2
  gam = min(gam, (p.x_max(i)-p.h_r(i))^2/Qinv(i,i));
end
for i = 1:2
  gam = min(gam, (p.h_r(i)-p.x_min(i))^2/Qinv(i,i));
end
Qisqrt = sqrtm(Qinv);
gam = 1.5*gam;
gamma_min = beta_cost/max(eig(Qisqrt*(pp.Q+K'*pp.R*K)*Qisqrt));
if gam < gamma_min
  error('Terminal constraint computation failed (not invariant: gamma = %.4e gamma_min = %.4e',gam,gamma_min);
end
V = Q_N/gam;
P = Q_N;

% V = Q_N/gam;
% P = Q_N;
