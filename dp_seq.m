function K = dp_seq(A,B,Q,R,P)

[nx,nu,N] = size(B);
K = zeros(nu,nx,N);

for k = N:-1:1
    A_ = A(:,:,k);
    B_ = B(:,:,k);

    S = B_'*P*B_ + R;
    BPA = B_'*P*A_;
    K(:,:,k) = -S\BPA; 
    P = Q + A_'*P*A_ - BPA'*(S\BPA);
end
