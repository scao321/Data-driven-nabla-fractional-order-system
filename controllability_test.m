clear
clc
w=foweight(-0.5,10);
n=2;% number of states
W =cell2mat( arrayfun(@(x) diag(x*ones(1,n)), w, 'UniformOutput', false));
A=[0,-1;1,-1];B=[0;1];%SIMO
rank(ctrb(A,B)) % check the controllability matrix
for j=1:2*n
X=[1,0,1,0
    0,1,1,0];%initial conditions
x=X(:,j);
U=[1,-1;-1,1;2,0;-1,1];
u=U(j,:);
for k=1:length(u)
    x(:,k+1)=A*x(:,k)+B*u(:,k)+(W(:,1:n*k)-W(:,n+1:n*(k+1)))*reshape(flip(x(:,1:k),2),[],1);
end
X1(:,j)=reshape(x(:,1:2),[],1);
X2(:,j)=reshape(x(:,2:3),[],1);
end
%%
syms l
matrix = X2- l * X1;
% Check the determinant
det(matrix)

%%
function w=foweight(alpha,L)
w=1;
for i=2:L
w(i)=w(i-1)*(1-(alpha+1)/(i-1));
end
end