#%% P154 zero coupon bond
def zero_coupon_bond(par, y, t):
    """
    Price a zero coupon bond.
    
    :param par: face value of the bond.
    :param y: annual yield or rate of the bond.
    :param t: time to maturity, in years.
    """
    return par/(1+y)**t

print(zero_coupon_bond(100, 0.05, 5))

#%% P162 Calculating the yield to maturity

import scipy.optimize as optimize

def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*2
    coupon = coup/100.*par
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y:sum([coupon/freq/(1+y/freq)**(freq*t) for t in dt]) 
    +  par/(1+y/freq)**(freq*T) - price
    
    return optimize.newton(ytm_func, guess)


ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
print(ytm)


#%% P163 Calculating the price of a bond

def bond_price(par, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T*2
    coupon = coup/100.*par
    dt = [(i+1)/freq for i in range(int(periods))]
    price = sum([coupon/freq/(1+ytm/freq)**(freq*t) for t in dt]) +  \
       par/(1+ytm/freq)**(freq*T)
    return price


price = bond_price(100, 1.5, ytm, 5.75, 2)
print(price)

#%% Bond duration
#percentage change in bond price with respect to a
#percentage change in yield , 1%)

def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    
    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    mduration = (price_minus-price_plus)/(2*price*dy)
    return mduration


mod_duration = bond_mod_duration(95.0428, 100, 1.5, 5.75, 2)
print(mod_duration)



#%% P165 Bond convexity

#Higher-convexity portfolios are less
#affected by interest-rate volatilities than lower-convexity portfolios, given the
#same bond duration and yield. As such, higher-convexity bonds are more
#expensive than lower-convexity ones, everything else being equal.


def bond_convexity(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    convexity = (price_minus + price_plus - 2*price)/(price*dy**2)
    return convexity

#%% P167 Short-rate modeling - The Vasicek model
# rates CAN become negative    
import math
import numpy as np

def vasicek(r0, K, theta, sigma, T=1., N=10, seed=777):    
    np.random.seed(seed) #to produce same results each time run
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])* dt + sigma*math.sqrt(dt)*np.random.normal()
        rates.append(rates[-1]+dr)
        
    return range(N+1), rates

#

# In[13]:
get_ipython().run_line_magic('pylab', 'inline')
fig = plt.figure(figsize=(12, 8))

for K in [0.002, 0.02, 0.2]:
    x, y = vasicek(0.005, K, 0.15, 0.05, T=10, N=200)
    plot(x,y, label='K=%s'%K)
    pylab.legend(loc='upper left');
    
pylab.legend(loc='upper left')
pylab.xlabel('Vasicek model');








