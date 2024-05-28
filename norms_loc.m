function vecnorm = norms_loc( x, p, dim )

if p == 0
    vecnorm = sqrt( sum( x .* conj( x ), dim ) );
else % if x is a 
    vecnorm = sqrt( sum( x .* transpose( x ), dim ) );
end
