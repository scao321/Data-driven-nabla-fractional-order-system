clear
clc 
n=2;% number of states
m=2;%number of inputs
T=eye(2);
A=[-0.8,1;-1,-0.3];B=[-1,2;1,1];
%%
k=6;% n(b)=n*k, lag<=n(b), lag*output>=n(b)
Tini=10;
Tf=10;
L=Tini+Tf;
k1=(m+1)*(L+n*k)-1;
alpha=0.1;
w=foweight(alpha-1,k1+1);
W =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), w, 'UniformOutput', false));
%%
for jj=1:200
u=randn(m,k1);
x0=randn(n,1);
x=trajectory(A,B,W,u,x0)+0*randn(2,k1);
wd=[u',x']; 
%%
%arx identification
data=iddata(wd(:,m+1:end),wd(:,1:m),1);
sys_id=arx(data,[k*ones(n,n),k*ones(n,n), zeros(n,n) ],'Ts',1);
sys=ss(sys_id);
y0=zeros(n*k,1);
[y,tsim]=lsim(sys,ones(m,100),[],y0);
%%
%slra
s.m=[k+1;k+1;k+1;k+1];
s.n=k1-k;
s.w=[inf*ones(1,m*k1),ones(1,n*k1)]';
opt.solver='c';
[ph, info] = slra([wd(:,1);wd(:,2);wd(:,3);wd(:,4)], s, m*(k+1)+n*k ,opt);
%%
  Ud1=[ph(1:k1),ph(k1+1:m*k1)];
  Yd1=[ph(m*k1+1:(m+1)*k1),ph((m+1)*k1+1:end)];
  %Ud1=u';
  %Yd1=x';

yff=dd_sim(u',x',Tini,Tf,m);
yff1=dd_sim(Ud1,Yd1,Tini,Tf,m);
%%
kt=100;
wt=foweight(alpha-1,kt+1);
Wt =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), wt, 'UniformOutput', false));
ut=ones(m,kt);
xt=trajectory(A,B,Wt,ut,[0;0]);
%%
err(jj)=sum(sum(abs(yff-xt),2)./sum(abs(xt),2));
err1(jj)=sum(sum(abs(yff1-xt),2)./sum(abs(xt),2));
err2(jj)=sum(sum(abs(y'-xt),2)./sum(abs(xt),2));
jj
end
%%
figure(1)
h = boxplot([log(err'),log(err1') log(err2')], 'Labels', {'original','approximated', 'arx'}, 'Widths', 0.5);
set(h,'LineWidth',2)
 set(h(3, :), 'LineWidth', 1.5);  % Box edges
 set(h(4, :), 'LineWidth', 1.5);  
ylabel('$\log(e)$', 'Interpreter', 'latex');
set(gca, 'FontSize', 15);
edges=linspace(0,1e-3,100);
figure(2)
histogram(err1,edges,Normalization="percentage")
hold on
histogram(err2,edges,Normalization="percentage")
legend('Approximated','ARX')
ytickformat("percentage")
xlabel('Residual error of simulation')
%%
figure(3)
subplot(2,1,1)
plot(xt(1,1:end),LineWidth=2)
hold on
plot(yff(1,1:end),'--o',LineWidth=2,MarkerIndices=1:5:length(yff))
plot(yff1(1,1:end),'--o',LineWidth=2,MarkerIndices=1:5:length(yff))
plot(y(1:end,1),'--',LineWidth=2)
subplot(2,1,2)
plot(xt(2,1:end),LineWidth=2)
hold on
plot(yff(2,1:end),'--o',LineWidth=2,MarkerIndices=1:5:length(yff))
plot(yff1(2,1:end),'--o',LineWidth=2,MarkerIndices=1:5:length(yff))
plot(y(1:end,2),'--',LineWidth=2)
legend('xr','xo','xd','xi');
%%
function yff=dd_sim(Ud1,Yd1,Tini,Tf,m)
[Up1,Uf1,~]=creatHankel(Ud1(:,1),Tini,Tf,1);
[Up2,Uf2,~]=creatHankel(Ud1(:,2),Tini,Tf,1);
[Yp1,Yf1,~]=creatHankel(Yd1(:,1),Tini,Tf,1);
[Yp2,Yf2,~]=creatHankel(Yd1(:,2),Tini,Tf,1);
 yini1=zeros(Tini,1);
 yini2=zeros(Tini,1);
 uini=zeros(m*Tini,1);
 uf=ones(m*Tf,1);
 yff=[];
 T=pinv([Yp1;Yp2;Up1;Up2;Uf1;Uf2]);
for r=1:ceil(100/Tf)
  g=T*[yini1;yini2;uini;uf];
  yf1=Yf1*g;
  yf2=Yf2*g;
if Tini>=Tf
    yini1=[yini1(Tf+1:end);yf1];
    yini2=[yini2(Tf+1:end);yf2];
    uini=[uini(m*Tf+1:end);uf];
else
    yini1=yf1(Tf-Tini+1:end);
    yini2=yf2(Tf-Tini+1:end);
    uini=[uf(m*(Tf-Tini)+1:end)];
end
yff=[yff,[yf1';yf2']];
end
end
%%
function s=trajectory(A,B,W,u,x0)
    [n,~]=size(A); [~,m]=size(B);
    x=x0;
    for k=1:length(u)
        x(:,k+1)=A*x(:,k)+B*u(:,k)+(W(:,1:n*k)-W(:,n+1:n*(k+1)))*reshape(flip(x(:,1:k),2),[],1);
    end
    s=x(:,2:end);
end

function w=foweight(alpha,L)
w=1;
for i=2:L
w(i)=w(i-1)*(1-(alpha+1)/(i-1));
end
end

function [Up,Uf,U1]=creatHankel(w,Tini,Tf,n)
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