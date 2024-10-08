function [x ci95 r2 p] = fit_linear(t,data,err,marker)
%[x ci95 r2 p] = fit_linear(t,data,[err],marker)
%
% Fits linear polynomial: data = x(1) + x(2)*t.
% Treats NaN as missing data.

% handle args
if nargin==2
    err = zeros(size(t));
    marker = 'o';
elseif nargin==3
    err = gather(err);
    if isnumeric(err)
        marker = 'o';
    elseif ischar(err)
        marker = err;        
        err = zeros(size(t));
    else
        error('argument 3 not compatible.')
    end
end
if numel(t)~=numel(err); error('t and err size mismatch'); end
if numel(t)~=numel(data); error('t and data size mismatch'); end

% discard NaN, reshape and sort ascending
t = reshape(t,[],1);
err = reshape(err,[],1);
data = reshape(data,[],1);

ok = find(isfinite(t) & isfinite(data));
t = t(ok);
err = err(ok);
data = data(ok);

[t,ok] = sort(t);
data = data(ok);
err = err(ok);

t = double(gather(t));
err = double(gather(err));
data = double(gather(data));

% regression, confidence intervals (95%), etc.
X = [ones(size(t)) t];
Y = data;
[x,BINT,R,RINT,STATS] = regress(Y,X);
ci95 = x-BINT(:,1);
r2 = STATS(1);
p = STATS(3);

%old way
% myfunc = @(x)x(1)+x(2)*t-data;
% [x,r2,~,~,~,~,J] = lsqnonlin(myfunc,[mean(data) 0]);
% v = numel(data)-numel(x); % deg. freedom
% cov = inv(full(J'*J))*r2/v;
% %ci95 = sqrt(diag(cov))*1.96;
% stderr = sqrt(diag(cov));
% ci95 = tinv(1-0.05/2,v)*stderr;


% display line
if any(err)
    errorbar(t,data,err,marker);
else
    plot(t,data,marker);
end
title(['x(1) + x(2)*t  (± 95% CI)'])
hold on
plot(t,x(1)+x(2)*t,'r');

% text box
str{1} = sprintf('slope = %.4f ± %.4f',x(2),ci95(2));
str{2} = sprintf('intercept = %.4f ± %.4f',x(1),ci95(1));
str{3} = sprintf('R-squared = %.4f (p=%.1e)',r2,p);
text(0.02,0.91,str,'Units','Normalized','FontName','FixedWidth')

% screen text
disp([' slope     ' num2str(x(2)) ' ± ' num2str(ci95(2)) ' (95% CI)'])
disp([' intercept ' num2str(x(1)) ' ± ' num2str(ci95(1)) ' (95% CI)'])
disp([' r2        ' num2str(r2,'%f') ' (n=' num2str(numel(data)) ')'])
disp([' p-value   ' num2str(p,'%e')])

% prediction interval
tpred = linspace(min(t),max(t),5*numel(t));
[lower upper] = predci(t,data,tpred);
plot(tpred,lower,'r:','linewidth',1)
plot(tpred,upper,'r:','linewidth',1)
hold off

%text(1.1*min(t),0.95*max(data),['slope     ' num2str(x(2),4) ' +/- ' num2str(ci95(2),3)])
%text(1.1*min(t),0.85*max(data),['intercept ' num2str(x(1),4) ' +/- ' num2str(ci95(1),3)])
%text(1.1*min(t),0.75*max(data),['r=' num2str(sqrt(r2),4) '  p=' num2str(p,'%1.0g')]);

drawnow


%% 95% prediction intervals for linear model y = b0 + b1*x at new locations xpred
function [ypred_lower ypred_upper] = predci(x,y,xpred)

% http://folk.ntnu.no/sveinoll/ov/TMA/4240%20Statistikk/H12/11%20Matlab/linearreg.m

% Calculate the mean of the given data
x_bar = mean(x);
y_bar = mean(y);

% Fit the model using least square method
Sxx = sum((x-x_bar).^2);
Sxy = sum((x-x_bar).*(y-y_bar));
Syy = sum((y-y_bar).^2);

% Calculate b1
b1 = Sxy/Sxx;

% Calcuate b0
b0 = y_bar - b1*x_bar;

% Compute the fitted y values for the values in x
y_hat = b0 + b1*x;

% Computes the residuals, e(i) = y(i) - yhat(i)
residuals = y-y_hat;

% Find the length of the data
n = length(x);

% Find the unbiased estimate of sigma^2
s = sqrt((Syy-b1*Sxy)/(n-2));

% Define the significance level
alpha = 0.05;

% Find the value for t-distribution with give values
t = tinv(1-alpha/2,n-2);

% 95% prediction interval 
ypred = b0+b1*xpred;
ypred_lower = ypred - t*s*sqrt(1+1/n+(xpred-x_bar).^2/Sxx);
ypred_upper = ypred + t*s*sqrt(1+1/n+(xpred-x_bar).^2/Sxx);
