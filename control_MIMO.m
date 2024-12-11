clear
clc 
% two inputs two outputs
m=2;
n=2;% number of states
A=[-0.8,1;-1,-0.3];
%B=[-.1,0.5;.1,.2]; %currently used in the paper, need to be changed later
B=[-1,2;1,1];
eig(A)
rank(ctrb(A,B))
%%
k=5;% n(b)=n*k, lag<=n(b), lag*output>=n(b)
Tini=5;
Tf=10;
L=Tini+Tf;
k1=(m+1)*(L+n*k)-1;
alpha=0.5;
w=foweight(alpha-1,k1+1);
W =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), w, 'UniformOutput', false));
%%
%linear system based on A,B
  % A11=zeros(n*k,n*k);
  % A1=eye(2);
  % cellArray = repmat({A1}, 1, k-1);
  % A11(1:end-2,3:end)=blkdiag(cellArray{:});
  % A11(end-1:end,:)=cell2mat( arrayfun(@(x) diag(x*ones(1,n)), flip(w(1:k)-[w(2:k),0]), 'UniformOutput', false));
  % A11(end-1:end,end-1:end)=A11(end-1:end,end-1:end)+A;
  % B1=[zeros(n*k-2,m);B];
  % C1=[zeros(2,n*k-2),eye(2)];
  % G=ss(A11,B1,C1,0,1);
  % y0=zeros(n*k,1);
  % [y,tsim]=lsim(G,ones(m,100),[],y0);
%%
rng(42)
u=randn(m,k1);
%[~,~,Urank]=creatHankel(u(:),Tini,Tf+n*k,m);
%rank(Urank)-(Tini+Tf+n*k)*m
x0=randn(n,1);
x=trajectory(A,B,W,u,x0)+0*randn(2,k1);
wd=[u',x']; 
%%
%arx identification
% data=iddata(wd(:,m+1:end),wd(:,1:m),1);
% sys_id=arx(data,[k*ones(n,n),k*ones(n,n), zeros(n,n) ],'Ts',1);
% sys=ss(sys_id);
%  y0=zeros(n*k,1);
%  [y,tsim]=lsim(sys,ones(m,100),[],y0);
%%
%slra
s.m=[k+1;k+1;k+1;k+1];
s.n=k1-k;
s.w=[inf*ones(1,m*k1),ones(1,n*k1)]';
%s.w=[inf*ones(1,m*k1),[inf*ones(1,k),ones(1,k1-k)],[inf*ones(1,k),ones(1,k1-k)]]';
opt.solver='c';
%opt.method='reg';
[ph, info] = slra([wd(:,1);wd(:,2);wd(:,3);wd(:,4)], s, m*(k+1)+n*k ,opt);
%%
  %Ud1=u';
  %Yd1=x';
   Ud1=[ph(1:k1),ph(k1+1:m*k1)];
   Yd1=[ph(m*k1+1:(m+1)*k1),ph((m+1)*k1+1:end)];
%%
% Ud1 is [u1;u2] Yd1 is [y1;y2]
[Up1,Uf1,U11]=creatHankel(Ud1(:,1),Tini,Tf,1);
[Up2,Uf2,U12]=creatHankel(Ud1(:,2),Tini,Tf,1);
[Yp1,Yf1,Y11]=creatHankel(Yd1(:,1),Tini,Tf,1);
[Yp2,Yf2,Y12]=creatHankel(Yd1(:,2),Tini,Tf,1);
%%
rank([U11;U12;Y11;Y12])-(m*(Tini+Tf)+n*k)
%digits(100);
%rank(vpa([U1;Y1]))-(m*(Tini+Tf)+n*k) % use vpa to convert to high precision

%%
kt=100;
wt=foweight(alpha-1,kt+1);
Wt =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), wt, 'UniformOutput', false));
 g = sdpvar(k1-Tini-Tf+1,1);
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
 T=pinv([Yp1;Yp2;Up1;Up2;Yf1;Yf2]);
 %%
for r=1:kt/steps
 objective =1*norm([yini1;yini2;up1;up2]-[Yp1;Yp2;Up1;Up2]*g,2);
 objective=objective+1*norm(yf1-Yf1*g,2)+1*norm(yf2-Yf2*g,2);
 objective=objective+0*norm(g,1)+0*norm(g,2);
 option=sdpsettings('solver','mosek');
 Constraints=[[Uf1;Uf2]*g<=1*ones(n*Tf,1)];
 Constraints=[Constraints,[Uf1;Uf2]*g>=-1*ones(n*Tf,1)];
 %Constraints=[];
 sol = optimize(Constraints,objective,option);
 solution=value(g);
 uf1=Uf1*solution;
 uf2=Uf2*solution;
   % g=T*[yini1;yini2;up1;up2;yf1;yf2];
   % uf1=Uf1*g;
   % uf2=Uf2*g;
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
% if r>10
%     yf1=-1*ones(Tf,1);
%     yf2=3*ones(Tf,1);
%     % yf1=floor(r*steps/10)*ones(Tf,1);
%     % yf2=-floor(r*steps/10)*ones(Tf,1);
% end
end

%%
subplot(2,1,1)
plot(0:99,yr2,LineWidth=2,Color='red');
hold on
plot(0:100,[0,xt(1,1:end)],"-.o",LineWidth=2,MarkerIndices=1:10:100,Color='blue')
plot(0:100,[0,xt(2,1:end)],"--*",LineWidth=2,MarkerIndices=5:10:100,Color='#77AC30')
%plot(yr1,'--',LineWidth=2);
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