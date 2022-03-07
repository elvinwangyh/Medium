# %% [markdown]
# # Regime Classification with Dynamic Voaltility Model

# %% [markdown]
# ## Overview

# %% [markdown]
# **Motivation:**   
# Volatility regime classification is a hot topic. so we will try out two models to classify them. 
#     
# **Objectives:**  
# 1. Dynamic volatility modeling with Generalized AutoRegressive Conditional Heteroskedasticity - GARCH(1,1)  
# 2. Volatility model parameter optimisation  
# 3. Volatility regime classification with Markov Models
# 4. Classification model performance evaluation  
#         
# **Flow of Analysis & Explanations:**  
# 1. Analysis of the return distribution: 
# > We will look at the higher moments of the return distribution.  
# > Then we will use statistical tests to determine if the distribution has 'zero mean' as well as if the distribution follows a normal distribution.  
# > These tests are important in determining the volatility model assumpotions, such as the mean model and distribution assumptions.  
# 2. GARCH(1,1) Modelling  
# > we will construct the volatility model based on the results we obtained above and optimise the parameters (Optimisation is embedded in the function).    
# > we will then compare the market volatility model against the VIX index to understand that the implied volatility will be different from conditional as the market may not assume risk neutrality.  
# 3. Hidden Markov Model 
# > Hidden markov model is one of the often mentioned perhaps due to the success of renaissance technology.  
# > We will examine the performance of the model, how to improve on the model and if the assumptions are actually correct.  
# 4. Markov Switching Autoregressive Model
# > This model was covered by a few papers.   
# > While the model performs better than HMM, the model has its own issues such as computation time and MLE convergence issue.  
#     
# **NOTE**  
# The legend in HMM can be wrongly labelled.
# 

# %% [markdown]
# ## Packages

# %%
# Packages
import pandas as pd 
import numpy as np
import datetime as dt 

from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
import arch

import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% [markdown]
# ## Data

# %%
# Data & processing

def get_return(ticker, period, interval):
    
    #############################
    # input: ticker and interval
    # output: return
    
    ticker = yf.Ticker(str(ticker))
    data = ticker.history(period = str(period), interval = str(interval))
    rt = data.Close.pct_change().dropna()
    
    return rt

# %%
rt = get_return(ticker = '^GSPC', period = 'max', interval = '1d')

# %% [markdown]
# ### Stylised facts of return distribution

# %%
def return_dist_stats(asset_return):
    
    ################################
    # Input: return series
    # output: stats and chart

    plt = px.histogram(asset_return, marginal= "violin")
    
    print('Stylised fact of the retrun distribution')
    print('-' * 50)
    print('Length of the return series:', asset_return.shape)
    print('Mean:', asset_return.mean()*100, '%')
    print('Standard Deviation:', asset_return.std()*100, '%')
    print('Skew:', asset_return.skew())
    print('Kurtosis:', asset_return.kurtosis())
    
    plt.show()

# %%
return_dist_stats(rt)

# %% [markdown]
# Based on the above statistics, we know the distribution is leaning to the positive side (negative skew) and narrow (positive kurtosis).

# %% [markdown]
# ### Hypothesis testing for mean return and distribution normality

# %%
def test_dist(asset_return, mode, alpha = 1e-2):
    ##############################
    # Input: return and test('mean' or 'normal'), alpha in decimals
    # output: P value and test result
    
    if (mode == 'mean'):
        t_stat, p = stats.ttest_1samp(asset_return, popmean=0, alternative='two-sided')
    elif (mode == 'normal'):
        k2, p = stats.normaltest(rt)
    
    def test(p, alpha):
        print("p = {:g}".format(p))
        if p < alpha: 
            print("The null hypothesis can be rejected")
        else:
            print("The null hypothesis cannot be rejected")
            
    return test(p, alpha)        
    

# %%
test_dist(rt, 'mean')

# %%
test_dist(rt, 'normal')

# %% [markdown]
# From the above testing, we can be confident that the return distribution has a non-zero mean and the return is not normally distributed.  
# Since the mean is non-zero, we will model the volatility with a conditional mean model, in this case hetergenous autoregressive model.  
# Using this mean model, the mean includes a component of an average of the mean during the lag period.  

# %% [markdown]
# ## Volatility Modelling

# %% [markdown]
# ### GARCH(1,1)
# The package ARCH will perform parameter optimisation of the model automatically.  
# since the return distribution is not normal, we can choose alternative distributions to model the volatility.  
# we are using heterogenous autoregressive mean and skewed student's t disrtibution based on maximum likelihood.  
# 

# %%
am = arch.univariate.arch_model(rt, x=None, 
                                mean='HARX', lags=0, 
                                vol='Garch', p=1, o=0, q=1, 
                                dist='skewt', hold_back=None, rescale=True)

volatility_model = am.fit()
volatility_model.summary()

# %% [markdown]
# ### Long-term variance under GARCH(1,1)

# %%
volatility_model.params

# %%
# Retrieve Model Parameters
const, omega, alpha, beta, eta, lamb = volatility_model.params

# Retrieve conditional volatility
garch_vol = volatility_model.conditional_volatility.round(2) * np.sqrt(252)

# long-term variance under GARCH
VL = omega / (1 - alpha - beta )

# long-term volatility under GARCH (convert from variance)
sigma_L = np.sqrt(VL) * np.sqrt(252) # already measured in percentage

# sample volatility estimate
sample_sigma = rt.std() *np.sqrt(252) * 100


# %%
# Volatlity Plot Function
def vol_plot(garch, vl, std):
    
    fig = px.line(garch,title="GARCH(1,1)")

    fig.add_hline(y=vl, line_dash="dash", line_color="green", annotation_text="Long-run variance estimate")

    fig.add_hline(y=std, line_dash="dash", line_color="red", annotation_text="Sample variance")


    fig.show()


# %% [markdown]
# ### Plot Everything

# %%
vol_plot(garch_vol, sigma_L, sample_sigma)

# %% [markdown]
# ### Comparison with VIX
# 

# %%
# Data & processing
ticker = yf.Ticker("^VIX")
vix = ticker.history(period = 'max', interval = "1d")
vix = vix.Close

val_data = pd.DataFrame([vix, garch_vol]).T.dropna()

# %%
px.line(val_data, line_shape='hv')

# %%
px.histogram(val_data.Close - val_data.cond_vol , marginal= "violin")

# %%
print('Mean Absolute Error: ', mean_absolute_error(val_data.iloc[:,0], val_data.iloc[:,1]))

test_dist(val_data.Close - val_data.cond_vol, 'mean')

# %% [markdown]
# So we know that the mean difference between model and market is significant.  
# However, VIX measures the implied volatility based on option price but our model measures the conditional volatility of the market.   
# These two are distinctly different and they are supposed to.     
# Implied higher than conditional volatility suggests the pricing of options may not be risk neutral.   

# %% [markdown]
# ## Hidden Markov Model

# %%
def fitHMM(vol, n_states):
    
    train_vals = np.expand_dims(vol, 1)
    
    train_vals = np.reshape(train_vals,[len(vol),1])
    
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=n_states, n_iter=100).fit(train_vals)
     
    # classify each observation as state 0 or 1
    hidden_states = model.predict(train_vals)
    post_prob = np.array(model.predict_proba(train_vals))
 
    # fit HMM parameters
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)
    print(mus)
    print(sigmas)
    
    relabeled_states = hidden_states
    return (relabeled_states, mus, sigmas, transmat, post_prob, model)

# %%
def plot_model(dates, vol, post_prob, export_label):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=vol, name="GARCH", mode='lines', line_shape='hv', yaxis = 'y1'))

    fig.add_trace(go.Scatter(x=dates, y=post_prob.iloc[:,0], name = 'Pr(Low Vol Regime)', mode='lines', line_shape='hv',
                             line=dict(width=0.5, color='green'), 
                             stackgroup='two', yaxis = 'y2'))

    fig.add_trace(go.Scatter(x=dates, y=post_prob.iloc[:,1], name = 'Pr(Medium Vol Regime)', mode='lines', line_shape='hv',
                             line=dict(width=0.5, color='orange'),
                             stackgroup='two', yaxis = 'y2'))

    fig.add_trace(go.Scatter(x=dates, y=post_prob.iloc[:,2], name = 'Pr(High Vol Regime)', mode='lines', line_shape='hv',
                             line=dict(width=0.5, color='red'),
                             stackgroup='two', yaxis = 'y2'))

    # Create axis objects
    fig.update_layout(
        title = ("Volatility Regime - " + str(export_label)),

        yaxis=dict(title="Volatility"),

        yaxis2=dict(title="Posterier Probability", overlaying="y1", side="right")

    )

    fig.write_html('Volatility Regime Classification - ' + str(export_label) + '.html') 

    fig.show()

# %%
hidden_states, mus, sigmas, transmat, post_prob, hmm_model = fitHMM(garch_vol, 3)
dates = garch_vol.index

hmm_data = pd.DataFrame([dates, garch_vol, hidden_states], 
                        index = ["date", "volatility", "hidden_states"]).T

hmm_prob = pd.DataFrame(post_prob, columns = ['state_1', 'state_2', 'state_3'])
hmm_data = pd.concat([hmm_data, hmm_prob], axis=1)

hmm_data.date = pd.to_datetime(hmm_data.date)
hmm_data = hmm_data.sort_values(by="date")

hmm_data

# %%
plot_model(hmm_data.date, hmm_data.volatility, hmm_prob, 'HMM')

# %% [markdown]
# ## Markov Switching Autoregression Model 

# %% [markdown]
# Intuitively, volatility regime change can be fast, yet the transition does not necessarily have to be instantaneous. The HMM model does not appear to be perform well under this intuition.   
# 

# %%
# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(rt-rt.mean(), k_regimes=3, order = 1, trend="n", switching_ar = False, switching_variance = True)
    
res_hamilton = mod_hamilton.fit()
res_hamilton.summary()

# %%
post_prob = res_hamilton.smoothed_marginal_probabilities
post_prob = pd.DataFrame(post_prob)

plot_model(dates, garch_vol, post_prob, 'MSAR')

# %% [markdown]
# ## Summary

# %%
hmm_log_prob = hmm_model.score(np.expand_dims(garch_vol,1))
print('log-likelihood of HMM:', hmm_log_prob)
print('Transition Matrix of MSAR:')
print(transmat)

# %%
msar_log_prob = mod_hamilton.loglike(res_hamilton.params)
trans_matrix = mod_hamilton.regime_transition_matrix(res_hamilton.params)
print('Log-likelihood of MSAR:', msar_log_prob)
print('Transition Matrix of MSAR:')
print(trans_matrix)

# %% [markdown]
# Hidden Markov Model has a lower log likelihood value than the Markov Switching Autoregressive Model,    

# %% [markdown]
# ## References  
# **By section:**  
# 1. Return distributional Assumptions  
# > ARCH Model https://www.fsb.miamioh.edu/lij14/672_2014_s5.pdf  
# > Heterogeneous Auroregressive Mean Model https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.HARX.html#arch.univariate.HARX  
# > Garch Forecasting Performance under Different Distribution Assumptions http://www-stat.wharton.upenn.edu/~steele/Courses/434/434Context/GARCH/Willhelmesson06.pdf  
# 
# 2. Volatility Modelling
# > Predicting volatility with heterogeneous autoregressive models https://www.sr-sv.com/predicting-volatility-with-heterogeneous-autoregressive-models/   
# 
# 
# 3. Hidden Markov Model
# > Practical Time Series Analysis - code repo https://github.com/PracticalTimeSeriesAnalysis/BookRepo      
# > HMMLearn https://hmmlearn.readthedocs.io/en/latest/
# > Quantstrat HMM https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/
# 
# 4. Markov Switching Autoregressive Model
# > ECB Volatility Regime https://www.ecb.europa.eu/pub/financial-stability/fsr/focus/2018/pdf/ecb~bcaaae16c3.fsrbox201805_03.pdf  
# > Autoregressive conditional heteroskedasticity and changes in regime https://www.sciencedirect.com/science/article/abs/pii/0304407694900671    
# > Markov-Switching - Kim, Nelson, and Startz (1998) Three-state Variance Switching http://www.chadfulton.com/topics/mar_kim_nelson_startz.html   
# > Statsmodels Variance Switching Model https://www.statsmodels.org/dev/examples/notebooks/generated/markov_autoregression.html#Kim,-Nelson,-and-Startz-(1998)-Three-state-Variance-Switching  
# > Statsmodels Markov Regression https://www.statsmodels.org/devel/examples/notebooks/generated/markov_regression.html   
# 


