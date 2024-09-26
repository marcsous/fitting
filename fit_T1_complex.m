function [T1 ci95] = fit_T1_complex(TI,data,TR,T1)
%[T1 ci95] = fit_T1_complex(TI,data,T1,TR)
% Least squares fitting of T1 to complex data.
%
% TI: vector of TIs (or scalar)
% data: matrix of data (multichannel in dim 2)
% TR: vector of TRs (or scalar)
% T1: scalar (initial estimate)
% ci95: 95% confidence interval
%
% Notes:
% TI/TR must be vector/scalar, scalar/vector or vector/vector.
%
%% parse arguments
if isvector(data)
    data = reshape(data,[],1);
elseif ndims(data)~=2
    error('data must be a vector or matrix');
end
[np nc] = size(data); % no. points, no. coils
if isscalar(TI)
    if numel(TR)~=np; error('TI/TR size mismatch'); end
    TI = repmat(TI,np,1);
end
if ~exist('TR','var') || isempty(TR)
    TR = Inf;
end
if isscalar(TR)
    TR = repmat(TR,np,1);
end
TI = reshape(TI,np,1);
TR = reshape(TR,np,1);

% sort to avoid weirdness
[~,k] = sortrows([TI TR]);
TI = TI(k); TR = TR(k); data = data(k,:);

%% initial estimate (R1 instead of T1)
if exist('T1','var') && ~isempty(T1)
    x = 1/T1;
else
    x = estimateR1(TI,data);
end

%% damped least squares
maxit = 20;

[r J y] = func(x,TI,TR,data);
lambda = real(J'*J);

for iter = 1:maxit

    % constrain to be real (doi.org/10.1016/j.laa.2010.07.011)
    dx = -real(J'*r) / real(J'*J+lambda);

    [s K z] = func(x+dx,TI,TR,data);

    if norm(s) < norm(r)
        x = x + dx;           
        r = s; J = K; y = z;     
        lambda = lambda / 2;
    else
        lambda = lambda * 3;
    end

end

% 95% confidence interval
df = numel(data)-(nc+1); % nc+1 parameters
err = (r'*r)/df;
cov = inv(J'*J)*err;
ci95 = sqrt(diag(cov))*1.96;

%% convert to T1
T1 = 1/x;
ci95 = ci95/x^2;

%% plot (won't run in parfor loop)
if isempty(gcp('nocreate'))
    phi = -exp(-i*angle(y(1,:)));
    h = plot(TI,real(phi.*data),'o');
    hold on
    for c = 1:nc
        plot(TI,real(phi(c)*y(:,c)),'Color',h(c).Color);
    end
    hold off
    xlabel('TI'); ylabel('Signal'); grid on;
    str = sprintf('T_1 = %.2f ± %.2f',T1,ci95);
    text(0.5,0.5,str,'units','normalized');
end

%% classic equation: y = 1-2exp(-TI*x)+exp(-TR*x)
function [r J y] = func(x,TI,TR,data)

f = 1 - 2*exp(-TI*x);
df = 2*TI.*exp(-TI*x);

if all(isfinite(TR))
    f = f + exp(-TR*x);
    df = df - TR.*exp(-TR*x);
end

A = (f'*data) / (f'*f); % VARPRO
dA = df'*(data-2*A.*f) / (f'*f);

y = A.*f;
r = y - data;
J = A.*df + dA.*f;

r = reshape(r,[],1);
J = reshape(J,[],1);

%% Barral model: y = a+b*exp(-TI*x)
% function [r J y] = func2(x,TI,TR,data)
% 
% f = exp(-TI*x);
% 
% A = [ones(size(f)) f]; % VARPRO
% pinvA = pinv(A);
% ab = pinvA*data;
% 
% y = A*ab;
% r = y - data;
% 
% df = -TI.*f;
% D = [zeros(size(df)) df];
% J = D*ab - A*pinv(A)*D*ab;
% 
% r = reshape(r,[],1);
% J = reshape(J,[],1);
%
%% linear estimation method: y = a + b*exp(c*x)
function c = estimateR1(x,y)
% github.com/juangburgos/FitSumExponentials/blob/main/matlab/02_With_Additive_Constant_Term.m
% modified for multiple RHS y = [y1 y2 y3...] with shared 'c'
[n nrhs] = size(y);

% perform integral
iy = cumtrapz(x,y);

% couple rhs (share index 1)
Y = zeros(n,nrhs,2*nrhs+1);
for k = 1:nrhs
    Y(:,k,1) = iy(:,k);
    Y(:,k,2*k) = x;
    Y(:,k,2*k+1) = 1;
end

% solve YA = y;
Y = reshape(Y,[],2*nrhs+1);
y = reshape(y,[],1);
A = pinv(Y) * y;

c = -real(A(1));