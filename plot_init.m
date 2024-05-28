function plot_init(x_0,u_0,x_r,V,p)

% Trajectory in phase plot
figure(1);
plot(x_0(1, :), x_0(2, :))
hold on 
rectangle('Position',[0 0 p.x_max(1) p.x_max(2)])
axis([p.x_min(1)-1 p.x_max(1)+1 p.x_min(2)-1 p.x_max(2)+1])
ellipse(V, x_r(:, end))
xlabel('$x_1$', 'Interpreter','latex')
ylabel('$x_2$', 'Interpreter','latex')
legend({'$x_0$', 'terminal set'}, 'Interpreter','latex')
grid on
figure(2);
stairs(u_0)
hold on;
grid on
