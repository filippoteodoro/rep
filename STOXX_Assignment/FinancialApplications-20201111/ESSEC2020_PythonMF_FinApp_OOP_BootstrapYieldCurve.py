#%% P158 Bootstrapping a yield curve
import math

class BootstrapYieldCurve():    
    
    def __init__(self):
        self.zero_rates = dict()
        self.instruments = dict()
        
    def add_instrument(self, par, T, coup, price, compounding_freq=2):
        self.instruments[T] = (par, coup, price, compounding_freq)
    
    def get_maturities(self):
        """ 
        :return: a list of maturities of added instruments 
        """
        return sorted(self.instruments.keys())
    
    def get_zero_rates(self):
        """ 
        Returns a list of spot rates on the yield curve.
        """
        self.bootstrap_zero_coupons()    
        self.get_bond_spot_rates()
        return [self.zero_rates[T] for T in self.get_maturities()]    
        
    def bootstrap_zero_coupons(self):
        """ 
        Bootstrap the yield curve with zero coupon instruments first.
        """
        for (T, instrument) in self.instruments.items():
            (par, coup, price, freq) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate        
                
    def zero_coupon_spot_rate(self, par, price, T):
        """ 
        :return: the zero coupon spot rate with continuous compounding.
        """
        spot_rate = math.log(par/price)/T
        return spot_rate
                    
    def get_bond_spot_rates(self):
        """ 
        Get spot rates implied by bonds, using short-term instruments.
        """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument
            if coup != 0:
                spot_rate = self.calculate_bond_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate
                
    def calculate_bond_spot_rate(self, T, instrument):
        try:
            (par, coup, price, freq) = instrument
            periods = T*freq
            value = price
            per_coupon = coup/freq
            for i in range(int(periods)-1):
                t = (i+1)/float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon*math.exp(-spot_rate*t)
                value -= discounted_coupon

            last_period = int(periods)/float(freq)        
            spot_rate = -math.log(value/(par+per_coupon))/last_period
            return spot_rate
        except:
            print("Error: spot rate not found for T=", t)
          


# In[3]:
yield_curve = BootstrapYieldCurve()
yield_curve.add_instrument(100, 0.25, 0., 97.5)
yield_curve.add_instrument(100, 0.5, 0., 94.9)
yield_curve.add_instrument(100, 1.0, 0., 90.)
yield_curve.add_instrument(100, 1.5, 8, 96., 2)
yield_curve.add_instrument(100, 2., 12, 101.6, 2)


# In[4]:
y = yield_curve.get_zero_rates()
x = yield_curve.get_maturities()
get_ipython().run_line_magic('pylab', 'inline')
fig = plt.figure(figsize=(12, 8))
plot(x, y)
title("Zero Curve") 
ylabel("Zero Rate (%)")
xlabel("Maturity in Years");

#%% P161 Forward rates
#fwd = r2T2-r1T1/T2-T1

class ForwardRates(object):
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()
        
    def add_spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate
        
    def get_forward_rates(self):
        """
        Returns a list of forward rates
        starting from the second time period.
        """
        periods = sorted(self.spot_rates.keys())
        for T2, T1 in zip(periods, periods[1:]):
            forward_rate = self.calculate_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)

        return self.forward_rates
    
    def calculate_forward_rate(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2*T2-R1*T1)/(T2-T1)
        return forward_rate        


# In[5]:
fr = ForwardRates()
fr.add_spot_rate(0.25, 10.127)
fr.add_spot_rate(0.50, 10.469)
fr.add_spot_rate(1.00, 10.536)
fr.add_spot_rate(1.50, 10.681)
fr.add_spot_rate(2.00, 10.808)

print(fr.get_forward_rates())
#[10.81, 10.603, 10.971, 11.189]


