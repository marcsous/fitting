function [ab ci95] = varpro(x,y,b,f,df)
%[ab ci95] = varpro(x,y,b,f,df)
%
% Least squares fitting: y = sum{a(n)*f{n}(x,b(n))}
%
% x: vector of coordinates
% y: vector of measurements
% b: initial estimates for b
% f: cell array of functions f(x,b)
% df: cell array of derivatives df(x,b)
%
% ab: parameters [linear nonlinear]
% ci95: 95% confidence intervals

%% example: y = a(1)*exp(b(1)*x) + a(2)*power(x,b(2));
if nargin==0
    x = logspace(0,2,10);
    f{1} = @(x,b) exp(b*x);
    f{2} = @(x,b) power(x,b);    
    df{1} = @(x,b) x.*exp(b*x);  
    df{2} = @(x,b) power(x,b).*log(x);
    y = 50*f{1}(x,-0.5) + 5*f{2}(x,0.5) + randn(size(x));
    b = [-1 1]; % needs reasonable initial estimates
end

%% parse arguments
n = numel(f); % no. functions
m = numel(x); % no. measurements
if numel(b)~=n; error('b and f size mismatch'); end
if numel(y)~=m; error('x and y size mismatch'); end
if numel(df)~=n; error('f and df size mismatch'); end
if ~isa(f,'cell'); error('f must be a cell array'); end
if ~isa(df,'cell'); error('df must be a cell array'); end

b = reshape(b,[],1);
x = reshape(x,[],1);
y = reshape(y,[],1);

%% damped least squares 
maxit = 20;

[r J] = func(b,x,y,f,df);

for iter = 1:maxit

    [U S V] = svd(J,'econ');
    if iter==1
        lambda = S(1)^2 / 2;
    end
    invS = S./(S.^2+lambda);
    db = -V * invS * U' * r;

    [s K] = func(b+db,x,y,f,df);

    if norm(s) < norm(r)
        b = b + db;     
        r = s; J = K;     
        lambda = lambda / 2;
    else
        lambda = lambda * 3;
    end

end

%% return arguments
[r J yhat a A] = func(b,x,y,f,df);

% linear and nonlinear parameters
ab(:,1) = a;
ab(:,2) = b;

% 95% confidence intervals
v = (r'*r)/(m-2*n);
cov = pinv(A'*A)*v;
ci95(:,1) = sqrt(diag(cov))*1.96;
cov = pinv(J'*J)*v;
ci95(:,2) = sqrt(diag(cov))*1.96

%% display
plot(x,y,'o');
hold on
plot(x,yhat);
hold off

%% generate fitting function
function [r J yhat a A] = func(b,x,y,f,df)

n = numel(f);
m = numel(x);

for k = 1:n
    A(:,k) = f{k}(x,b(k));
end
pinvA = pinv(A);

a = pinvA*y;
yhat = A*a;
r = yhat - y;

for k = 1:n
    D = df{k}(x,b(k));
    J(:,k) = D*a(k) - A*pinvA*D*a(k);
end
