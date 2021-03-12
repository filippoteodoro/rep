'''
Getting data from Yahoo Finance
Getting data from Quandl
Getting data from Intrinio
Converting prices to returns
Changing frequency
Visualizing time series data
Identifying outliers
Investigating stylized facts of asset returns
'''
#%%============================================================================
#                 Downloading data
#==============================================================================

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# In[3]:
import matplotlib.pyplot as plt
import warnings
plt.style.use('seaborn')
# plt.style.use('seaborn-colorblind') #alternative
# plt.rcParams['figure.figsize'] = [16, 9]
#plt.rcParams['figure.dpi'] = 300
warnings.simplefilter(action='ignore', category=FutureWarning)

# In[1]:Import the libraries:
import pandas as pd 
import yfinance as yf

# In[2]:Download the data:
#We can pass a list of multiple tickers, such as ['AAPL', 'MSFT'].
#We can set auto_adjust=True to download only the adjusted prices.
#We can additionally download dividends and stock splits by
#setting actions='inline'.
#Setting progress=False disables the progress bar.

# can also use library called yahoofinancials

df_yahoo = yf.download('AAPL', 
                       start='2000-01-01', 
                       end='2010-12-31',
                       progress=False)
# In[3]:Inspect the data:
print(f'Downloaded {df_yahoo.shape[0]} rows of data.')
df_yahoo.head()

# In[4]:Getting data from Quandl 
import pandas as pd 

# In[34]:2. Authenticate using the personal API key
QUANDL_KEY = "gM7v-yxiuRqvkW71yqZh" # replace {key} with your own API key  
quandl.ApiConfig.api_key = QUANDL_KEY

# In[11]:# 3. Download the data:
df_quandl = quandl.get(dataset='WIKI/AAPL',
                       start_date='2000-01-01', 
                       end_date='2010-12-31')

#The collapse parameter can be used to define the frequency (available options:
#daily, weekly, monthly, quarterly, or annually).

# In[12]: # 4. Inspect the data:
print(f'Downloaded {df_quandl.shape[0]} rows of data.')
df_quandl.head()

# In[5]: Getting data from Intrinio
import intrinio_sdk
import pandas as pd

# In[6]:# 2. Authenticate using the personal API key and select the API:

intrinio_sdk.ApiClient().configuration.api_key['api_key'] = '{key}'  # replace {key} with your own API key  
security_api = intrinio_sdk.SecurityApi()

# In[7]:# 3. Request the data:

r = security_api.get_security_stock_prices(identifier='AAPL', 
                                           start_date='2000-01-01',
                                           end_date='2010-12-31', 
                                           frequency='daily',
                                           page_size=10000)

# In[8]:# 4. Convert the results into a DataFrame:
# Intrio returns a JSON like object

response_list = [x.to_dict() for x in r.stock_prices]
df_intrinio = pd.DataFrame(response_list).sort_values('date')
df_intrinio.set_index('date', inplace=True)

# In[9]:# 5. Inspect the data:
print(f'Downloaded {df_intrinio.shape[0]} rows of data.')
df_intrinio.head()

#%% Other Libraries
#iexfinance: A library that can be used to download data from IEX Cloud
#tiingo: A library that can be used to download data from Tiingo
#alpha_vantage: A library that is a wrapper for the Alpha Vantage API

#%%============================================================================
#                   Converting prices to returns
#==============================================================================
import pandas as pd 
import numpy as np
import yfinance as yf
# In[36]:2. Download the data and keep the adjusted close prices only:
df = yf.download('AAPL', 
                 start='2000-01-01', 
                 end='2010-12-31',
                 progress=False)
#only takes Adj Close column
df = df.loc[:, ['Adj Close']]
#renames column
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)

# In[37]:3. Convert adjusted close prices to simple and log returns:

df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

# In[38]:# 4. Inspect the output:
df.head()

# In[24]:# 1. Using QUANDL
import pandas as pd
import quandl
#conda install -c anaconda quandl
# https://docs.quandl.com/docs/python-time-series
QUANDL_KEY =  "gM7v-yxiuRqvkW71yqZh" # replace {key} with your own API key  
quandl.ApiConfig.api_key = QUANDL_KEY
 
# In[49]:2. Create a DataFrame with all possible dates and left join 
#the prices on it, and forward fill the missing dates prices:
df_all_dates = pd.DataFrame(index=pd.date_range(start='1999-12-31', 
                                              end='2010-12-31'))

#left join returns all rows from the left table and the matched rows from the right table
#while leaving the unmatched rows empty

df = df_all_dates.join(df[['adj_close']], how='left').fillna(method='ffill').asfreq('M')

# In[50]:3. Download inflation data from Quandl:
df_cpi = quandl.get(dataset='RATEINF/CPI_USA', 
                    start_date='1999-12-01', 
                    end_date='2010-12-31')
df_cpi.rename(columns={'Value':'cpi'}, inplace=True)

# In[51]:4. Merge inflation data to prices:
df_merged = df.join(df_cpi, how='left')

# In[52]:5. Calculate simple returns and inflation rate:

df_merged['simple_rtn'] = df_merged.adj_close.pct_change()
df_merged['inflation_rate'] = df_merged.cpi.pct_change()


# In[53]:# 6. Adjust returns for inflation by column:
df_merged['real_rtn'] = (df_merged.simple_rtn + 1) / (df_merged.inflation_rate + 1) - 1
df_merged.head()

#%%============================================================================
#                   Changing frequency of volatility
#==============================================================================
# 0. Obtain the simple returns in case of starting in this recipe:
import pandas as pd;import yfinance as yf;import numpy as np
# download data 
df = yf.download('AAPL', 
                 start='2000-01-01', 
                 end='2010-12-31', 
                 auto_adjust=False,
                 progress=False)

# keep only the adjusted close price
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# calculate simple returns
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

# remove redundant data, drop adj_close and na
df.drop('adj_close', axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

df.head()

# In[4]:# 1. Import the libraries:
import pandas as pd 

# In[5]:# 2. Define the function for calculating the realized volatility:
def realized_volatility(x):
    return np.sqrt(np.sum(x**2))

# In[6]:# 3. Calculate monthly realized volatility:

df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
df_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)

# In[7]:# 4. Annualize the values:
df_rv.rv = df_rv.rv * np.sqrt(12)

# In[10]:# 5. Plot the results:

#spikes in the realized volatility coincide with some extreme
#returns
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df)
ax[1].plot(df_rv)
plt.tight_layout()
# plt.savefig('images/ch1_im6.png')
plt.show()


#%% Can also use resample to calculate
#average monthly return

#df.log_rtn.resample('M').mean()

#%%============================================================================
#           Visualizing time series data                
#==============================================================================
import pandas as pd 
import yfinance as yf

# download data as pandas DataFrame
df = yf.download('MSFT', auto_adjust = False, progress=False)
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# create simple and log returns
df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

# dropping NA's in the first row
df.dropna(how = 'any', inplace = True)

# In[14]:the `plot` method of pandas

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
# add prices
df.adj_close.plot(ax=ax[0])
ax[0].set(title = 'MSFT time series',
          ylabel = 'Stock price ($)')
    
# add simple returns 
df.simple_rtn.plot(ax=ax[1])
ax[1].set(ylabel = 'Simple returns (%)')

# add log returns 
df.log_rtn.plot(ax=ax[2])
ax[2].set(xlabel = 'Date', 
          ylabel = 'Log returns (%)')

ax[2].tick_params(axis='x', 
                  which='major', 
                  labelsize=12)

# plt.tight_layout()
# plt.savefig('images/ch1_im7.png')
plt.show()

#%%In[14]: `plotly` + `cufflinks`
#more interactive BUT only in Jupyter Notebook

# 1. Import the libraries and handle the settings:
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode

# set up settings (run it once)
# cf.set_config_file(world_readable=True, theme='pearl', 
#                    offline=True)
# initialize notebook display
init_notebook_mode()

# In[15]: # 2. Create the plots:
df.iplot(subplots=True, shape=(3,1), shared_xaxes=True, title='MSFT time series')


#%%============================================================================
#            Identifying outliers                
#==============================================================================
import pandas as pd 
import yfinance as yf
df = yf.download('AAPL', 
                 start='2000-01-01', 
                 end='2010-12-31',
                 progress=False)

df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)

# In[5]:
df['simple_rtn'] = df.adj_close.pct_change()

# In[6]:
df.head()

# In[7]:# 1. Calculate the rolling mean and standard deviation:
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])

#col names from (simple_rtn,mean) --> mean , std
#drops a level ie simple_rtn
df_rolling.columns = df_rolling.columns.droplevel()


# In[8]:# 2. Join the rolling metrics to the original data:
df_outliers = df.join(df_rolling)

# In[9]:# 3. Define a function for detecting outliers:
def indentify_outliers(row, n_sigmas=3):
    '''
    Function for identifying the outliers using the 3 sigma rule. 
    The row must contain the following columns/indices: simple_rtn, mean, std.
    
    Parameters
    ----------
    row : pd.Series
        A row of a pd.DataFrame, over which the function can be applied.
    n_sigmas : int
        The number of standard deviations above/below the mean - used for detecting outliers
        
    Returns
    -------
    0/1 : int
        An integer with 1 indicating an outlier and 0 otherwise.
    '''
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1
    else:
        return 0 

# In[10]:# 4. Identify the outliers and extract their values for later use:
df_outliers['outlier'] = df_outliers.apply(indentify_outliers,  axis=1)

#returns simple_rtn col where col outlier = 1
outliers = df_outliers.loc[df_outliers['outlier'] == 1, ['simple_rtn']]

# In[11]:# 5. Plot the results: 
#when there are two large returns in the vicinity, the algorithm identifies the
#first one as an outlier and the second one as a regular observation. This might be
#due to the fact that the first outlier enters the rolling window and affects the
#moving average/standard deviation.

#Outlier-Isolation Forest, Hampel Filter, Support Vector Machines, and z-score

fig, ax = plt.subplots()
ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
ax.set_title("Apple's stock returns")
ax.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('images/ch1_im9.png')
plt.show()
#%%============================================================================
#    Investigating stylized facts of asset returns                
#==============================================================================
# 1. Import the libraries:
import pandas as pd 
import numpy as np
import yfinance as yf
import seaborn as sns 
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
# In[14]:# 2. Download the S&P 500 data and calculate the returns:
df = yf.download('^GSPC', 
                 start='1985-01-01', 
                 end='2018-12-31',
                 progress=False)
df = df[['Adj Close']].rename(columns={'Adj Close': 'adj_close'})
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df = df[['adj_close', 'log_rtn']].dropna(how = 'any')



# In[15]:Fact 1 - Non-Gaussian distribution of returns
#Negative skewness (third moment): Large negative returns occur more
#frequently than large positive ones.
#Excess kurtosis (fourth moment) : Large (and small) returns occur more often
#than expected

# 1. Calculate the Normal PDF using the mean and standard deviation of the 
#observed returns:
r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)   

# In[16]:# 2. Plot the histogram and the Q-Q Plot:
#fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fig, ax = plt.subplots(1, 2)
# histogram -> sns.distplot while setting kde=False (which does not use
#the Gaussian kernel density estimate) and norm_hist=True (this plot shows density
#instead of the count).
sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])                                    
ax[0].set_title('Distribution of MSFT returns', fontsize=16)                                                    
ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].legend(loc='upper left');

# Q-Q plot
#means that the left tail of the returns distribution is heavier than
#that of the Gaussian distribution
qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize = 16)

plt.tight_layout()
# plt.savefig('images/ch1_im10.png')
plt.show()

# In[66]: # 3. Print the summary statistics of the log returns:

#the Jarque-Bera normality test gives us reason to reject
#the null hypothesis stating that the distribution is normal at the 99% confidence
#level.

#With a pvalue of zero, we reject the null hypothesis that sample data has skewness and kurtosis
#matching those of a Gaussian distribution.


jb_test = scs.jarque_bera(df.log_rtn.values)

print('---------- Descriptive Statistics ----------')
print('Range of dates:', min(df.index.date), '-', max(df.index.date))
print('Number of observations:', df.shape[0])
print(f'Mean: {df.log_rtn.mean():.4f}')
print(f'Median: {df.log_rtn.median():.4f}')
print(f'Min: {df.log_rtn.min():.4f}')
print(f'Max: {df.log_rtn.max():.4f}')
print(f'Standard Deviation: {df.log_rtn.std():.4f}')
print(f'Skewness: {df.log_rtn.skew():.4f}')
print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}') 
print(f'Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}')


# In[17]:# #### Fact 2 - Volatility Clustering
# 1. Run the following code to visualize the log returns series:
df.log_rtn.plot(title='Daily MSFT returns', figsize=(10, 6))
plt.tight_layout()
# plt.savefig('images/ch1_im12.png')
plt.show()

# In[16]:# #### Fact 3 - Absence of autocorrelation in returns
# 1. Define the parameters for creating the Autocorrelation plots:
N_LAGS = 50
SIGNIFICANCE_LEVEL = 0.05

#Run the following code to create ACF plot of log returns:
acf = smt.graphics.plot_acf(df.log_rtn, 
                            lags=N_LAGS, 
                            alpha=SIGNIFICANCE_LEVEL)

plt.tight_layout()
# plt.savefig('images/ch1_im13.png')
plt.show()

#Only a few values lie outside the confidence interval (we do not look at lag 0) and
#can be considered statistically significant. We can assume that we have verified
#that there is no autocorrelation in the log returns series.


# In[18]:Fact 4 - Small and decreasing autocorrelation in squared/absolute returns

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS, 
                      alpha=SIGNIFICANCE_LEVEL, ax = ax[0])
ax[0].set(title='Autocorrelation Plots',
          ylabel='Squared Returns')

smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS, 
                      alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
ax[1].set(ylabel='Absolute Returns',
          xlabel='Lag')

plt.tight_layout()
# plt.savefig('images/ch1_im14.png')
plt.show()



# In[21]:Fact 5 - Leverage effect
# 1. Calculate volatility measures as moving standard deviations
# asset's volatility are negatively correlated
# with its returns, so price drops, vol spikes

df['moving_std_252'] = df[['log_rtn']].rolling(window=252).std()
df['moving_std_21'] = df[['log_rtn']].rolling(window=21).std()


# In[23]:# 2. Plot all the series:
#fig, ax = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
fig, ax = plt.subplots(3, 1, sharex=True)

df.adj_close.plot(ax=ax[0])
ax[0].set(title='MSFT time series', ylabel='Stock price ($)')

df.log_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Log returns (%)')

df.moving_std_252.plot(ax=ax[2], color='r',label='Moving Volatility 252d')
df.moving_std_21.plot(ax=ax[2], color='g', label='Moving Volatility 21d')
ax[2].set(ylabel='Moving Volatility',  xlabel='Date')
ax[2].legend()

plt.tight_layout()
# plt.savefig('images/ch1_im15.png')
plt.show()


# In[24]: Leverage Effect - using VIX
#Download and preprocess the prices of S&P 500 and VIX:

df = yf.download(['^GSPC', '^VIX'], 
                 start='1985-01-01', 
                 end='2018-12-31',
                 progress=False)
df = df[['Adj Close']]
df.columns = df.columns.droplevel(0)
df = df.rename(columns={'^GSPC': 'sp500', '^VIX': 'vix'})


# In[25]:# 2. Calculate log returns:
df['log_rtn'] = np.log(df.sp500 / df.sp500.shift(1))
df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
df.dropna(how='any', axis=0, inplace=True)

# In[26]:# 3. Plot a scatterplot with the returns on the axes and fit a regression line to identify trend:
corr_coeff = df.log_rtn.corr(df.vol_rtn)

ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, 
                 line_kws={'color': 'red'})
ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})',
       ylabel='VIX log returns',
       xlabel='S&P 500 log returns')

plt.tight_layout()
# plt.savefig('images/ch1_im16.png')
plt.show()

