clear
clc 
% two inputs two outputs
m=2;
n=2;% number of states
A=[-0.8,1;-1,-0.3];
B=[-1,2;1,1];
%%
k=5;% n(B)=n*k, lag<=n(B), lag*output>=n(B)
Tini=5;
Tf=10;
L=Tini+Tf;
k1=(m+1)*(L+n*k)-1;
alpha=0.5;
w=foweight(alpha-1,k1+1);
W =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), w, 'UniformOutput', false));
%%
rng(42) 
u=randn(m,k1);
x0=randn(n,1);
x=trajectory(A,B,W,u,x0)+0*randn(2,k1);
wd=[u',x']; 
%%
%slra [1] I. Markovsky and K. Usevich, "Software for weighted structured low-rank approximation," J. Comput. Appl. Math., vol. 256, pp. 278–292,2014
s.m=[k+1;k+1;k+1;k+1];
s.n=k1-k;
s.w=[inf*ones(1,m*k1),ones(1,n*k1)]';
opt.solver='c';
[ph, info] = slra([wd(:,1);wd(:,2);wd(:,3);wd(:,4)], s, m*(k+1)+n*k ,opt);
%%
Ud1=[ph(1:k1),ph(k1+1:m*k1)];
Yd1=[ph(m*k1+1:(m+1)*k1),ph((m+1)*k1+1:end)];
%%
% Ud1 is [u1;u2] Yd1 is [y1;y2]
[Up1,Uf1,U11]=creatHankel(Ud1(:,1),Tini,Tf,1);
[Up2,Uf2,U12]=creatHankel(Ud1(:,2),Tini,Tf,1);
[Yp1,Yf1,Y11]=creatHankel(Yd1(:,1),Tini,Tf,1);
[Yp2,Yf2,Y12]=creatHankel(Yd1(:,2),Tini,Tf,1);
%%
kt=100;
wt=foweight(alpha-1,kt+1);
Wt =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), wt, 'UniformOutput', false));
g = sdpvar(k1-Tini-Tf+1,1); % yalmip setup [2] J. Lofberg, "YALMIP : A toolbox for modeling and optimization in matlab," in In Proceedings of the CACSD Conference, Taipei, Taiwan, 2004
 yini1=zeros(Tini,1);
 yini2=zeros(Tini,1);
 up1=zeros(Tini,1);
 up2=up1;
yf1=1*ones(Tf,1);
yf2=1*ones(Tf,1);
uff=[];
steps=1;
yr1=[];
yr2=[];
 %%
for r=1:kt/steps  
 objective =1*norm([yini1;yini2;up1;up2]-[Yp1;Yp2;Up1;Up2]*g,2);
 objective=objective+1*norm(yf1-Yf1*g,2)+1*norm(yf2-Yf2*g,2);
 objective=objective+0*norm(g,1)+0*norm(g,2);
 option=sdpsettings('solver','mosek'); 
 Constraints=[[Uf1;Uf2]*g<=1*ones(n*Tf,1)];
 Constraints=[Constraints,[Uf1;Uf2]*g>=-1*ones(n*Tf,1)];
 sol = optimize(Constraints,objective,option);
 solution=value(g);
 uf1=Uf1*solution;
 uf2=Uf2*solution;
  uff=[uff,[uf1(1:steps)';uf2(1:steps)']];
  xt=trajectory(A,B,Wt,uff,[0;0]);
if steps<=Tini
    yini1=[yini1(1+steps:end);xt(1,end-steps+1:end)'];
    yini2=[yini2(1+steps:end);xt(2,end-steps+1:end)'];
    up1=[up1(1+steps:end);uf1(1:steps)];
    up2=[up2(1+steps:end);uf2(1:steps)];
else
    yini1=xt(1,end-Tini+1:end)';
    yini2=xt(2,end-Tini+1:end)';
    up1=uf1(1:Tini);
    up2=uf2(1:Tini);
end
yr1=[yr1;yf1(1:steps)];
yr2=[yr2;yf2(1:steps)];
end

%%
subplot(2,1,1)
plot(0:99,yr2,LineWidth=2,Color='red');
hold on
plot(0:100,[0,xt(1,1:end)],"-.o",LineWidth=2,MarkerIndices=1:10:100,Color='blue')
plot(0:100,[0,xt(2,1:end)],"--*",LineWidth=2,MarkerIndices=5:10:100,Color='#77AC30')
legend('Reference','x_1','x_2')
grid on
ylabel('Outputs','FontSize',15)
subplot(2,1,2)
stairs(0:99,uff(1,1:end),LineWidth=2,Color='blue')
hold on
stairs(0:99,uff(2,1:end),LineWidth=2,Color='#77AC30')
legend('U_1','U_2')
xlabel('Time','FontSize',15)
ylabel('Inputs','FontSize',15)
grid on
%%
function s=trajectory(A,B,W,u,x0)
    [n,~]=size(A); [~,m]=size(u);
    x=x0;
    for k=1:m
        x(:,k+1)=A*x(:,k)+B*u(:,k)+(W(:,1:n*k)-W(:,n+1:n*(k+1)))*reshape(flip(x(:,1:k),2),[],1);
    end
    s=x(:,2:end);
    %s=x;
end
%%
function w=foweight(alpha,L)
w=1;
for i=2:L
w(i)=w(i-1)*(1-(alpha+1)/(i-1));
end
end
%%
function [Up,Uf,U1]=creatHankel(w,Tini,Tf,n)
%n numbers
U1=[];
[r,col]=size(w);
L=Tini+Tf;
d=r/n-L+1;
for i=1:col
    U=[];
    wd=w(:,i);
    for ii=1:L*n
    for j=1:d
        U(ii,j)=wd(ii+(j-1)*n);
    end
    end
    U1=[U1,U];
end
Up=U1(1:n*Tini,:);
Uf=U1(n*Tini+1:end,:);
end