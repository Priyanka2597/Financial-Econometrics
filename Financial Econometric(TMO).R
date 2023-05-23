
rm(list = ls())
library("rugarch")
library(quantmod)
library(fBasics)
library(forecast)
library(fBasics)
library(moments)
library(tseries)
library(stats)
library(TSA)
library("zoom")
library(MTS)

#Chosen Stock Thermo Fisher Scientific
#Download TMO stock prices
tickr <- "TMO"
start_date <- as.Date("1980-01-01")
end_date <- as.Date("2023-04-13")

# Use the getSymbols function to retrieve the data
getSymbols(tickr, src = "yahoo", from = start_date, to = end_date)
tmo_d <- Ad(TMO)
tmo_d <- na.omit(tmo_d)

#Calculate simple returns
tmo_d_sr <- diff(tmo_d)/tmo_d[-length(tmo_d)]
tmo_d_sr <- na.omit(tmo_d_sr)
tmo_log <- log(tmo_d)
tmo_log <- na.omit(tmo_log)
plot(tmo_log)

#Calculate log returns
tmo_d_lr <- diff(log(tmo_d))
tmo_d_lr <- na.omit(tmo_d_lr)

#Converting prices and returns into time series object
price_d <- ts(tmo_d)
ret_d <- ts(tmo_d_lr)

#Basic statistical examination of prices and returns
library(fBasics)
basicStats(tmo_d)
basicStats(tmo_d_lr)


#Perform skewness and kurtosis tests on returns
sk_p = skewness(ret_d);
T <- length(ret_d); 
tst = abs(sk_p/sqrt(6/T))
pv <- 2*(1-pnorm(tst))
pv

kt_p <- kurtosis(ret_d)
tst <- abs(kt_p/sqrt(24/T))
pv <- 2*(1-pnorm(tst))
pv


#This is just done for our understanding to observe the pattern in price series
##Unit root test on prices
#Perform test for mean price being zero
t.test(price_d)

#Perform normality test using the Jaque-Bera method
normalTest(price_d, method="jb")

#Ljung box test 
log(T)
Box.test(price_d, lag = 5, type = "Ljung")

#This is done for our analysis and included in the report
##Unit root test on Returns
#Perform test for mean return being zero
t.test(ret_d)

#Perform normality test using the Jaque-Bera method
normalTest(ret_d, method="jb")

#Ljung box test 
log(T)
Box.test(ret_d, lag = 10, type = "Ljung")

#Perform the Dickey-Fuller test on prices and returns
adf.test(price_d)
adf.test(ret_d)

#ACF and PACF of return
par(mfrow=c(1,2))
acf(ret_d)
pacf(ret_d)
## ACF and PACF suggests lag 1, lag 3
########DONE####

#EACF
library(forecast)
library(TSA)

#Returns (no need for differencing)
eacf(ret_d, ar.max = 7, ma.max = 13)
#(0,2),(0,4), (1,4),(1,2), (1,3), (2,3)

##Return models##

#Checking for various ARIMA model log return series
m1 = auto.arima(ret_d)
m2 = Arima(ret_d, order = c(1,0,2))
m3 = Arima(ret_d, order = c(1,0,4))
m4 = Arima(ret_d, order = c(1,0,3))
m5 = Arima(ret_d, order = c(2,0,3)) 
m6 = Arima(ret_d, order = c(0,0,4))
#Chosen model based on AIC

#dataframe of AIC
aic_p <- list(m1$aic, m2$aic, m3$aic, m4$aic, m5$aic, m6$aic)
print(aic_p)

#Residual Checking
#Check Residuals
#library(stats)

checkresiduals(m1, lag=10)
checkresiduals(m2, lag=10)
checkresiduals(m3, lag=10)
checkresiduals(m4, lag=10)
checkresiduals(m5, lag=10) #choosen model 
checkresiduals(m6, lag=10) 

#Ljung box test for residuals
Box.test(m5$residuals, lag = 10, type = "Ljung")
#Box.test(m6$residuals, lag = 10, type = "Ljung")

#Forecasting returns
forecast_r <- forecast(m5, h=100,level=c(90,95))
plot(forecast_r)
##### how to put red box and then done

###ARCH and GARCH###
library("zoom")
library(moments)
library(tseries)
library(stats)
library("rugarch")
#9. Checking for ARCH effect
r = m5$residuals
#9A. ACF and PACF of ARIMA(2,0,3) residuals to check for heteroscedasticity problem
par(mfrow=c(2,2))
acf(abs(r), main="ACF of absolute residuals")
pacf(abs(r), main="PACF of absolute residuals")
acf(r^2, main="ACF of squared residuals")
pacf(r^2, main="PACF of squared residuals")
# ACF and PACF plots of absolute and squared residuals show significant spikes 
# i.e. significant lag correlation

#9B. Testing for ARCH effect
at = r - mean(r)
# at is the mean-differenced series of residuals of ARMIMA (2,0,3) model fitted
# on Log.Return series
# at^2 is the series of variance of the residuals of the ARIMA model.

# Generalized Portmanteau Test for the ARCH effect
Box.test(at^2,lag=9,type="Ljung")
# H0: No ARCH effect
# p<0.05 : The variance of residuals have some autocorrelation 
# i.e. there is ARCH effect. 

archTest(r)
# p<0.05 : Presence of ARCH effect

#9C. Determining GARCH model 
# Cannot directly determine GARCH model from ACF and PACF, so we use EACF
eacf(abs(r))
# Suggests GARCH: (1,1), (1,2), (2,3), (3,3)
eacf(r^2)
# No clear suggestions

g11=garch(r,order=c(1,1))
g12=garch(r,order=c(1,2))
g23=garch(r,order=c(2,3))
g33=garch(r,order=c(3,3))
aic_garch <- list(AIC(g11), AIC(g12), AIC(g23), AIC(g33))

print(aic_garch)
# Lowest AIC is for GARCH (1,1) model.

summary(g11)
# All estimates are significant for the GARCH (1,1) model and it has the lowest 
# AIC, thus it provide the best fit

# Checking model
Box.test(g11$residuals, lag=9, type = "Ljung")
# p>0.05 : No autocorrelation in residuals

#9D. Fitting GARCH(1,1) model on residuals of ARIMA(2,0,3) on Log.Return series 
# checking normality assumption for residuals of GARCH(1,1) 
qqnorm(residuals(g11)); qqline(residuals(g11))
# Plot of residuals suggests student-t distribution

#Using rugarch package to fit the GARCH model
ug_spec = ugarchspec(mean.model = list(armaOrder=c(2,3)), distribution.model="std")
# Default is standard ARMA(1,1), so changed the order to ARMA (2,3)
# Default is Standard GARCH (1,1), so no need to change the order of GARCH
fit1 = ugarchfit(spec = ug_spec, data = ret_d)
# Checking for sign bias in the fitted model
fit1
# Sign bias is significant: There is leverage effect - Volatility reacts
# differently for positive and negative jumps so going for EGarch and Tgarch

#9E. Fitting eGARCH model
#Changing distributions to student-t in the above models

ug_spec6 = ugarchspec(variance.model=list(model="fGARCH",submodel="TGARCH", 
                                          garchOrder=c(1,1)), mean.model=list(armaOrder=c(2,3), 
                                                                              include.mean=TRUE),  distribution.model="std")  
fit6 = ugarchfit(spec = ug_spec6, data = ret_d)
fit6

ug_spec7 = ugarchspec(variance.model=list(model="eGARCH", garchOrder=c(1,1)),
                      mean.model=list(armaOrder=c(2,3), include.mean=TRUE),  
                      distribution.model="std") 
fit7 = ugarchfit(spec = ug_spec7, data = ret_d)
fit7


ug_spec9 = ugarchspec(variance.model=list(model="eGARCH", garchOrder=c(1,1)), 
                      mean.model=list(armaOrder=c(2,3), include.mean=TRUE, archm = TRUE),  
                      distribution.model="std" )
fit9 = ugarchfit(spec = ug_spec9, data = ret_d)
fit9
# ArchM parameter- p>0.05 : Not significant

par(mfrow=c(1,1))
plot(fit6,which=3) 
# Grey line is the plot of the series. Blue line represents the volatility 
# of the series.

plot(fit6,which=9)
# Qqplot

plot(fit7,which=3)
plot(fit7,which=9)

# Forecasting
boot_t_garch=ugarchboot(fit6,method=c("Partial","Full")[1],n.ahead = 12,
                        n.bootpred=1000,n.bootfit=1000)
plot(boot_t_garch,which=3) #Sigma Forecast
plot(boot_t_garch,which=2) #Log.Returns Forecast

boot_e_garch=ugarchboot(fit7,method=c("Partial","Full")[1],n.ahead = 12,
                        n.bootpred=1000,n.bootfit=1000)
plot(boot_e_garch,which=3) #Sigma Forecast
plot(boot_e_garch,which=2) #Log.Returns Forecast


##VAR models###
library(quantmod)
library(ggplot2)
library(MTS)
library(zoo)
library(fGarch)

# Set the start and end dates as Date type
start_date <- as.Date("1980-01-01")
end_date <- Sys.Date()

#Download daily prices for Tmo and S&P 500 using quantmod
getSymbols(c("TMO", "^GSPC"), src = "yahoo", from = start_date, to = end_date)

# Extract the 'Close' column from the data for Tmo and S&P 500
Tmo_prices <- data.frame(Date = index(TMO), Close = as.numeric(TMO$TMO.Close))
sp500_prices <- data.frame(Date = index(GSPC), Close = as.numeric(GSPC$GSPC.Close))

# Merge the data into a single data frame
merged_prices <- merge(Tmo_prices, sp500_prices, by = "Date")

plot(sp500_prices$Close)

# Taking the log of prices since prices are non-stationary
Tmo_prices$log_price <- log(Tmo_prices$Close)
sp500_prices$log_price <- log(sp500_prices$Close)

# Merge the data into a single data frame
merged_prices <- merge(Tmo_prices, sp500_prices, by = "Date")

# Create a line plot with two lines, one for each stock
ggplot(merged_prices, aes(x = Date)) +
  geom_line(aes(y = log_price.x), color = 'blue') +
  geom_line(aes(y = log_price.y), color = 'red') +
  
  # Set the title and y-axis label
  labs(title = "Log Prices of Tmo and S&P 500", y = " Log Price ($)") +
  
  # Adjust x-axis labels
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Extract the log prices from the merged dataframe
log_prices <- log(merged_prices[,2:3])

# Replace missing values with 0
log_prices_filled <- na.fill(log_prices, 0)

# Taking the first difference.
diff_prices <- diffM(log_prices_filled)

# Cross-correlation matrices
ccm(diff_prices)
#Order Specification
VARorder(diff_prices)
m1=VAR(diff_prices,p = 13) ### Fit a VAR(13) model
m1$coef
m2=VAR(diff_prices,p = 6) ### Fit a VAR(6) model
m2$coef
m3=VAR(diff_prices,p = 9) ### Fit a VAR(9) model
m3$coef

### Model checking
MTSdiag(m1, gof = 24, adj = 0, level = F)
MTSdiag(m2, gof = 24, adj = 0, level = F)
MTSdiag(m3, gof = 24, adj = 0, level = F)



###VALUE AT RISK ANALYSIS###

library(quantmod)
library(xts)
library(tseries)


##Download apple stock prices
tickr <- "TMO"
sdate <- '1980-01-01'
edate <- '2023-04-13'

#Download daily stock price and remove NA values
getSymbols(tickr, src = "yahoo", from = sdate, to = edate)
tmo_d <- Ad(TMO)
tmo_d <- na.omit(tmo_d)

#Download monthly stock price and remove NA values
getSymbols(tickr, src = "yahoo", from = sdate, to = Sys.Date(), 
           periodicity = 'monthly')
tmo_m <- Ad(TMO)
tmo_m <- na.omit(tmo_m)

#Calculate log returns for daily and monthly prices
tmo_d_lr <- diff(log(tmo_d))
tmo_d_lr <- na.omit(tmo_d_lr)

tmo_m_lr <- diff(log(tmo_m))
tmo_m_lr <- na.omit(tmo_m_lr)

#Converting the price and returns into time series
price_d <- ts(tmo_d)
price_m <- ts(tmo_m, frequency=12, start=c(1980,1))

ret_d <- ts(tmo_d_lr)
ret_m <- ts(tmo_m_lr, frequency=12, start=c(1980,1))



tmo=log(ret_d+1)
ntmo=-tmo    ## loss value
head(ntmo)



##RISK MODELLING
getwd()

### RiskMetrics #########
source("RMfit.R")
RMfit(ntmo)
### One can use default parameter beta = 0.96 wihtout estimation with the command
RMfit(ntmo,estim=F)


### Econometric modeling
library(fGarch)
m1=garchFit(~garch(1,1), data = tmo,trace=FALSE)
summary(m1)
pm1=predict(m1,10)
pm1
source("RMeasure.R")
RMeasure(-.00162,.01415)

#### 10-day VaR
names(pm1)
v1=sqrt(sum(pm1$standardDeviation^2))
RMeasure(-0.00162,v1)
m2=garchFit(~garch(1,1),data=tmo,trace=F,cond.dist="std")
summary(m2)
pm2=predict(m2,1)
pm2
RMeasure(.001491,.0139,cond.dist="std",df=5.066)

### Empirical quantile and quantile regression
quantile(ntmo,c(0.95,0.99,0.999))
library(quantmod)
library(tidyverse)
start_date <- as.Date("1980-01-01")
end_date <- as.Date("2023-04-13")
tickers <- c("TMO", "^GSPC")
getSymbols(tickers, from = start_date, to = end_date)

colnames(TMO)

# Remove the last row of the GSPC and AAPL
GSPC <- head(GSPC, -53)
TMO <- head(TMO$TMO.Volume, -1)

nrow(TMO$TMO.Volume)

tmo_data <- as.data.frame(cbind(ntmo, TMO$TMO.Volume))
GSPC_data <- as.data.frame(`GSPC`)
head(GSPC_data)

df <- cbind(tmo_data, GSPC_data$GSPC.Adjusted)
colnames(df) <- c("ntmo", "vol", "GSPC")

library(quantreg)
require(quantreg)

# fit quantile regression model
m3=rq(ntmo~vol+GSPC,data=df,tau=0.95)
summary(m3)
ts.plot(ntmo)

vol_last <- as.numeric(tail(df$vol, n = 1))
GSPC_last <- as.numeric(tail(df$GSPC, n = 1))
VaR_quant <- 0.03188 + 0*vol_last + 0.00000*GSPC_last
lines(m3$fitted.values,col="red")
VaR_quant

# fit quantile regression model
m4=rq(ntmo~vol+GSPC,data=df,tau=0.99)
summary(m4)
ts.plot(ntmo)

vol_last <- as.numeric(tail(df$vol, n = 1))
GSPC_last <- as.numeric(tail(df$GSPC, n = 1))
VaR_quant <- 0.05498 + 0*vol_last - 0.00001*GSPC_last
lines(m4$fitted.values,col="red")
VaR_quant

# fit quantile regression model
m5=rq(ntmo~vol+GSPC,data=df,tau=0.999)
summary(m5)
ts.plot(ntmo)

vol_last <- as.numeric(tail(df$vol, n = 1))
GSPC_last <- as.numeric(tail(df$GSPC, n = 1))
VaR_quant <- 0.12691 + 0*vol_last - 0.00001*GSPC_last
lines(m5$fitted.values,col="red")
VaR_quant