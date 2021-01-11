#==============================================================================
#                   EUROPEAN CALL OPTION PAYOFF
#==============================================================================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family']='serif'

# Option Strike
K = 8000

# Graph
S = np.linspace (7000,9000,100)  # linspace (start,stop, Returns num (100) evenly spaced samples, calculated over the interval [start, stop].)
C =np.maximum(S-K,0)

plt.figure()
plt.plot(S,C)
plt.xlabel('Index Level $S_t$ at maturity')
plt.ylabel('Intrinsic Value of call option')
plt.grid(True)

#%%============================================================================
#                    BLACK SCHOLES FOR EUROPEAN CALL OPTION
#==============================================================================
#%%
import math
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
'''
This is a module to calculate xxxxxxx
Inputs
S - stock price - float
K - stike price - float
T - time to maturity int /float
Outputs
C - option price - float
'''
#def calculate_d1(St, K, t, T, r, sigma):
#    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
#    return d1    

# =============================================================================
# def dN(x):
#     'pdf of a standard normal random variable x'
#     return math.exp (- 0.5 * x ** 2) / math.sqrt(2 * math.pi)
#           
# def N(d):
#     'cdf of a standard normal random variable x'
#     return quad (lambda x: dN(x), - 20 , d , limit = 50)[0]  #quad(func, lower limit which is -infinity, upper limit, An upper bound on the number of subintervals used in the adaptive algorithm =50)
# =============================================================================

def BSM_put_value(St, K, t, T, r, sigma):
     'Calculates Black Scholes Merton European put option value by using put call parity'
     put_value = BSM_call_value(St, K, t,T, r,sigma) - St + math.exp(-r * (T-t)) * K 
     return put_value

def BSM_call_value(St, K, t, T, r, sigma):
     d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
     d2 = d1 - sigma * math.sqrt(T)
     
     call_value = St * norm.cdf(d1, 0.0, 1.0) - K * math.exp(-r *T) * norm.cdf(d2, 0.0, 1.0) # an alternate way to calculate it is to directly use cdf function
     
     
     #calculate_d1(St, K, t, T, r, sigma)
     #d2 = d1 - sigma * math.sqrt(T)  
     #call_value = St * N(d1) - K * math.exp(-r * T) * N(d2)
     return call_value
#%%   
St = 100
K = 100
t = 0
T = 1
r = 0.05
sigma = 0.2
print ("Value of call option using Black Scholes is %f" % BSM_call_value(St, K, t, T, r, sigma)) # %f is assigned the float value
print ("Value of put option using Black Scholes is %f" % BSM_put_value(St, K, t, T, r, sigma))

#%%
def plotgraphs(x, y, xlabel, OptionType):
   plt.figure()
   plt.figure(figsize = (3,3)) 
   plt.plot(x,y)
   plt.grid(True)
   plt.xlabel(xlabel)
   plt.ylabel('Present Value(PV)')
   plt.title(OptionType + ' option versus ' + xlabel)  
 
#%% Call vs Strike  
# the further away K is , ie out of money , the lower the PV is   
   
points = 100 # points is used in np.linspace function below
klist = np.linspace(80,120, points)
vlist = [BSM_call_value(St, K, t, T, r, sigma) for K in klist] # ensure vlist is an array
plotgraphs(klist,vlist,'K','Call')


#%% =============================================================================
#                               CALCULATION OF GREEKS
# =============================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#from BSM_option_valuation import N,dN   # for some reason import also gives BS value
# http://www.macroption.com/option-greeks-excel/     # for formulaes
def dN(x):
    'pdf of a standard normal random variable x'
    return math.exp (- 0.5 * x ** 2) / math.sqrt(2 * math.pi)
          
def N(d):
    'cdf of a standard normal random variable x'
    return quad (lambda x: dN(x), - 20 , d , limit = 50)[0]  #quad(func, lower limit which is -infinity, upper limit, An upper bound on the number of subintervals used in the adaptive algorithm =50)

def BSM_delta(St, K, t, T, r,sigma): # Delta
    'Delta for European call option'
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
    delta = N(d1)
    return delta

# Gamma
def BSM_gamma(St, K, t,T, r,sigma):
    'Delta for European call option'
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
    gamma = dN(d1)/(St * sigma * math.sqrt(T))
    return gamma
    
# Theta
def BSM_theta(St, K, t,T, r,sigma):
    'Delta for European call option'
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    theta = (-St * dN(d1) * sigma)/ (2 * math.sqrt(T)) - (r * math.exp(-r* T) * K * N(d2))
    return theta    
     # double check formula as code says +     
    
def BSM_rho(St, K, t,T, r,sigma):
    'Rho for European call option'
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T- t)
    rho = K * (T) * math.exp(-r*T) * N(d2)
    return rho
           
def BSM_vega(St, K, t,T, r,sigma):
    'Vega for European call option'
    d1 = (math.log(St/K) + (r + 0.5 * sigma **2) * T)/(sigma * math.sqrt(T))
    vega = St * dN(d1) * math.sqrt(T)
    return vega
#%%
St = 100
K = 100
t = 0
T = 1
r = 0.05
sigma = 0.2

print ("Delta of call option using Black Scholes is %f" % BSM_delta(St, K, t, T, r, sigma)) 
print ("Gamma of call option using Black Scholes is %f" % BSM_gamma(St, K, t, T, r, sigma)) 
print ("Theta of call option using Black Scholes is %f" % BSM_theta(St, K, t, T, r, sigma)) 
print ("Rho of call option using Black Scholes is %f" % BSM_rho(St, K, t, T, r, sigma)) 
print ("Vega of call option using Black Scholes is %f" % BSM_vega(St, K, t, T, r, sigma)) 
#%% 
def plot_greeks(x, y, xlabel, ylabel,OptionType):
   plt.figure()
   plt.figure(figsize = (3,3)) 
  #plt.subplot(subplot)   
   plt.plot(x,y)
   plt.grid(True)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.title(OptionType + ' ' +  ylabel + ' vs ' + xlabel)  
#%% 
# Call Delta vs Spot
points = 100 # points is used in np.linspace function below
slist = np.linspace(40,150, points)
deltalist = [BSM_delta(St, K, t, T, r, sigma) for St in slist] # ensure vlist is an array
plot_greeks(slist,deltalist,'S','Delta', 'Call')  
    
# Call Delta vs Strike   
klist = np.linspace(40,150, points)
deltalist = [BSM_delta(St, K, t, T, r, sigma) for K in klist] # ensure vlist is an array
plot_greeks(klist,deltalist,'K','Delta', 'Call')  
#
  # Call Delta vs T
tlist = np.linspace(0.0001,1, points)
deltalist = [BSM_delta(St, K, t, T, r, sigma) for T in tlist] # ensure vlist is an array
plot_greeks(tlist, deltalist,'T','Delta', 'Call')  
      
  # Call Gamma vs Spot
points = 100 # points is used in np.linspace function below
slist = np.linspace(0.1,200, points)
gammalist = [BSM_gamma(St, K, t, T, r, sigma) for St in slist] # ensure vlist is an array
plot_greeks(slist,gammalist,'S','Gamma', 'Call') 
  
  # Call Gamma vs T
tlist = np.linspace(0.0001,1, points)
gammalist = [BSM_gamma(St, K, t, T, r, sigma) for T in tlist] # ensure vlist is an array
plot_greeks(tlist, gammalist,'T','Gamma', 'Call') 
    
   # Call Vega vs Spot
points = 100 # points is used in np.linspace function below
slist = np.linspace(40,150, points)
vegalist = [BSM_vega(St, K, t, T, r, sigma) for St in slist] # ensure vlist is an array
plot_greeks(slist,vegalist,'S','Vega', 'Call') 
  
  # Call Vega vs T
tlist = np.linspace(0.0001,1, points)
vegalist = [BSM_vega(St, K, t, T, r, sigma) for T in tlist] # ensure vlist is an array
plot_greeks(tlist, vegalist,'T','Vega', 'Call')  
    
   # Call Theta vs Spot
points = 100 # points is used in np.linspace function below
slist = np.linspace(40,200, points)
thetalist = [BSM_theta(St, K, t, T, r, sigma) for St in slist] # ensure vlist is an array
plot_greeks(slist,thetalist,'S','Theta', 'Call') 
  
  # Call Vega vs T
tlist = np.linspace(0.0001,1, points)
thetalist = [BSM_theta(St, K, t, T, r, sigma) for T in tlist] # ensure vlist is an array
plot_greeks(tlist, thetalist,'T','Theta', 'Call')


#%%==============================================================================
#                      MONTE CARLO SIMULATION FOR EUROPEAN CALL OPTION
#==============================================================================
import math, time
#from scipy import stats
from random import gauss, seed
seed(20000)

S0 = 100
K = 105
T = 1
r = 0.05
sigma = 0.2
M = 50 # number of timesteps
dt = T/M
I = 10 # number of paths
#  Simulate I paths with M timesteps
S = []
for i in range(I):
      path = []
      for t in range( M + 1): # range creates one less value eg range(3) = 0,1,2
          if t==0:
              path.append(S0)
          else:
              z = gauss(0.0, 1.0)
              St = path[t - 1] * math.exp((r- 0.5 * sigma ** 2) * dt  + sigma * math.sqrt(dt) * z) # why must use path[-1] ????
              path.append(St)   # path is the simulated value for one run of the stock 
      
      S.append(path)        # S is the list which contains eg 10 runs ! MUST CONVERT
#%% Calculate Monte Carlo estimator 
# First Way:Using a longer for loop
sum_val = 0.0
for path in S:
      sum_val += max(path[-1] - K, 0)     # use path[-1] to get the final stock value in one path
Method1C0 = math.exp(- r * T) * sum_val/I

print('European Option Value using Method 1 is %f' %round(Method1C0,3))
#%% Second Way using a for loop but using  List Comprehension
Method2C0 = math.exp(- r* T) * sum([max(path[-1] - K, 0) for path in S])/ I   # for each individual path in the S vector, take the max(final stock value -K) and add up all the values
print('European Option Value using Method 2 is %f' %round(Method2C0,3)) 
#%%
#Example 3-3 Monte Carlo Simulation using NumPy vector approach
import numpy as np
import math, time
np.random.seed(20000)
#t0 = time() #start time
S0 = 100; K = 105; T = 1; r = 0.05; sigma = 0.2; M = 50 # number of timesteps
dt = T/M; I = 10 

# simulate I paths with M time steps
S = np.zeros((M + 1, I)) # so effectively the x-axis becomes number of runs while y-axis becomes number of timesteps
S[0] = S0  # this makes the 0th row all S0 = 100


for t in range(1, M + 1): #range(1,51) = 1,2,3...50
    z = np.random.standard_normal(I) # at each time step eg t=1 ,simulate all the random numbers and hence S1 for all 250k paths, then go to next row t=2 and again simulate horizontally all 250k values
   
    # very important to specify np.exp else by default it will take math.exp and it wont work for arrays     
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt  + sigma * math.sqrt(dt) * z)

# very important to use np.sum and np.maximum
VectorMethod3C0 = math.exp(- r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I  # takes the entire last row ie the t=50th row as a vector, subtracts K and averages the entire row

#tpy3 = time() - t0
type(S)
type(z)
print('European Option Value using Vector Method 3 is %7.3f' %round(VectorMethod3C0,3))
#print('Time taken to price is %7.3f' %tpy3)

 # <codecell>
import matplotlib.pyplot as plt
plt.plot(S[:, :10])             # wrong as this is a dataframe plots the first 10 path
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')


#%%============================================================================
#  Estimating Value-at-risk using Monte Carlo
#==============================================================================
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns

# In[43]:# set random seed for reproducibility
np.random.seed(42)

# In[44]:# 2. Define the parameters that will be used for this exercise:
RISKY_ASSETS = ['GOOG', 'FB']
SHARES = [5, 5]
START_DATE = '2018-01-01'
END_DATE = '2018-12-31'
T = 1
N_SIMS = 10 ** 5

# In[45]:# 3. Download data from Yahoo Finance:

df = yf.download(RISKY_ASSETS, start=START_DATE, 
                 end=END_DATE, adjusted=True)
print(f'Downloaded {df.shape[0]} rows of data.')


# In[46]:
df.head()
# In[47]:# 4. Calculate daily returns:
adj_close = df['Adj Close']
returns = adj_close.pct_change().dropna()
plot_title = f'{" vs. ".join(RISKY_ASSETS)} returns: {START_DATE} - {END_DATE}'
returns.plot(title=plot_title)

plt.tight_layout()
plt.savefig('images/ch6_im3.png')
plt.show()

print(f'Correlation between returns: {returns.corr().values[0,1]:.2f}')

# In[48]:# 5. Calculate the covariance matrix:
cov_mat = returns.cov()
cov_mat

# In[49]:# 6. Perform the Cholesky decomposition of the covariance matrix:
chol_mat = np.linalg.cholesky(cov_mat)
chol_mat


# In[50]:# 7. Draw correlated random numbers from Standard Normal distribution:
rv = np.random.normal(size=(N_SIMS, len(RISKY_ASSETS)))
correlated_rv = np.transpose(np.matmul(chol_mat, np.transpose(rv)))

# In[51]:# 8. Define metrics used for simulations:
r = np.mean(returns, axis=0).values
sigma = np.std(returns, axis=0).values
S_0 = adj_close.values[-1, :]
P_0 = np.sum(SHARES * S_0)

# In[52]:# 9. Calculate the terminal price of the considered stocks:

S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + 
                   sigma * np.sqrt(T) * correlated_rv)

# In[53]:# 10. Calculate the terminal portfolio value and calculate the portfolio returns:
P_T = np.sum(SHARES * S_T, axis=1)
P_diff = P_T - P_0

# In[54]:# 11. Calculate VaR:
P_diff_sorted = np.sort(P_diff)
percentiles = [0.01, 0.1, 1.]
var = np.percentile(P_diff_sorted, percentiles)

for x, y in zip(percentiles, var):
    print(f'1-day VaR with {100-x}% confidence: {-y:.2f}$')

# In[55]:# 12. Present the results on a graph:

ax = sns.distplot(P_diff, kde=False)
ax.set_title('''Distribution of possible 1-day changes in portfolio value 
             1-day 99% VaR''', fontsize=16)
ax.axvline(var[2], 0, 10000)

plt.tight_layout()
plt.savefig('images/ch6_im4.png')
plt.show()

# In[56]:There's more

var = np.percentile(P_diff_sorted, 5)
expected_shortfall = P_diff_sorted[P_diff_sorted<=var].mean()

print(f'The 1-day 95% VaR is {-var:.2f}$, and the accompanying Expected Shortfall is {-expected_shortfall:.2f}$.')

