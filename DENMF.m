function [U_f,V_f,A_f,Z] = DENMF(W,k,traindata,trainlabel,testdata,lambda,alpha)


% Input:
% W ... weight matrix of LLE
% k ... number of classes
% testdata ... unlabeled data samples
% traindata & trainlabel ... labeled data samples and corresponding labels
% lambada & alpha ... trade-off parameter

% Optimization problem:
% \|X-U(AZ)^T\|^2_F+lambda*tr(A^T(I-W^T)(I-W)A)+alpha*tr[(AZ)^T(I-W^T)(I-W)AZ]
% s.t. U\ge 0, A\ge 0,Z\ge 0, A_L=F_L.

% Notation:
% X ... (mFea x nSmp) data matrix

% References:
% [1] Wenhui Wu, Sam Kwong, Junhui Hou, Yuheng Jia, Horace Ip, "Simultaneous 
%     Dimensionality Reduction and Classification via Dual Embedding 
%     Regularized Nonnegative Matrix Factorization", IEEE Transactions on Image 
%     processing, 2019.



maxIter=300;

X=[traindata;testdata]';
X=X+0.00000001*ones(size(X,1),size(X,2));
m=size(X,1);% number of features
n=size(X,2);% number of samples
nl=length(trainlabel);% number of labeled samples
I=eye(n);

Os=ones(n,1);
nIter=0;
num=1;

lle=(I-W')*(I-W);
iw=I+W'*W;
ww=W+W';

tryNo=0;
obj=1e+7;

while tryNo < 10 
    tryNo = tryNo+1;
    
    % ------initialize A
    A = zeros(n,k);
    A(nl+1:end,:)=NormalizeK(rand(n-nl,k));
    for i=1:nl
        A(i,trainlabel(i))=1;
    end
    
    % ------initialize U Z
    U=rand(m,k);
    [U] = NormalizeK(U);
    Z=rand(size(A,2),k);
    [Z] = NormalizeK(Z);

   

while nIter<maxIter
    
    % -----update U
    
    C=X*A*Z;
    D=U*Z'*A'*A*Z;
    U=U.*(C./D);

    
    % -----update A

    E=X'*U*Z'+lambda*(ww*A)+alpha*(ww*A*Z*Z');
    F=A*Z*U'*U*Z'+lambda*iw*A+alpha*iw*A*Z*Z';
    A=A.*(E./F);
    
    for i=1:nl
        A(i,trainlabel(i))=1;
    end
    % -----update Z

    G=A'*X'*U+alpha*(A'*ww*A*Z);
    H=A'*A*Z*U'*U+alpha*(A'*iw*A*Z);
    Z=Z.*(G./H);
    
    nIter=nIter+1;
    
end

V=A*Z;
FR=norm(X-U*Z'*A')+lambda*trace(A'*lle*A)+alpha*trace(Z'*A'*lle*A*Z);

if FR < obj
    A_f = A;
    V_f = V;
    U_f = U;
    obj=FR;
end
nIter=0;


end
[U_f] = NormalizeK(U_f);

   
function [K] = NormalizeK(K)
n = size(K,2);
norms = max(1e-15,sqrt(sum(K.^2,1)))';
K = K*spdiags(norms.^-1,0,n,n);


