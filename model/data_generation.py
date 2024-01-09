import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import Dataset

def generate_brownian_motion(S_0,T, N, mu, sigma, seed=None):
    """
    Generate 1-dimensional Brownian motion.

    Parameters:
    - T: Total time
    - N: Number of time steps
    - mu: Drift coefficient
    - sigma: Diffusion coefficient (volatility)
    - seed: Seed for reproducibility (optional)

    Returns:
    - t: Time array
    - W: Brownian motion array
    """

    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    dW = sigma * np.random.normal(0, np.sqrt(dt), N)

    W = np.zeros(N + 1)
    W[1:] = np.cumsum(dW)

    # Adding drift
    drift_one_time =(mu -(sigma**2)/2)*dt
    drift = drift_one_time*t
    W = S_0 * np.exp(W + drift)
    return t, W


# Example usage:
S_0 = 245
T = 101 # Total time in days
N = T*4  # Number of time steps
mu = 0.0212 # Drift coefficient
sigma = 0.38  # Diffusion coefficient (volatility)
seed = 42  # Seed for reproducibility
r_f = 0.0212 # real interest rate as a decimal
K = 294

t, W = generate_brownian_motion(S_0, T, N, mu, sigma, seed)

# Plotting the Brownian motion
plt.plot(t, W)
plt.title('Simulated Stock price for S_0=' + str(S_0))
plt.xlabel('Days')
plt.ylabel('Price')
#plt.show()

def black_scholes_call_price(a):
    """
    solves the black scholes equation (european call) using the classical formulation
    - x[0] = S_0: stock price at start date
    - x[1] = K: Strike price of the call option
    - x[2] = sigma: Volatility of the stock option, for now assume it is fixed.
    - x[3] = T: time to maturity in days
    - x[4] = r_f: riskfree rate, in year units.
    """
#internal parameters used to solve the equation
    d_1 = (torch.log(a[0] / a[1]) + (a[3]/365)*(a[4] + 0.5*x[2]**2))/ (a[2] * torch.sqrt(a[3]/365) )
    d_2 = d_1 - a[2] * torch.sqrt(a[3]/365)
#black scholes calculation
    c = a[0] * norm.cdf(d_1) - a[1] * np.exp(-a[4] * a[3]/365)*norm.cdf(d_2)
    return torch.tensor(c)

# general format: x = torch.tensor([S_0, K, sigma, T, r_f])

x = torch.tensor([S_0, K, sigma, T, r_f])

data_size = 10000
vector_dim = 5

data = torch.zeros(data_size, vector_dim)
#fill data with randomly generated values.

#fill in initial stock stock price
data[:, 0] = torch.rand(data_size)*300

#fill in strike price data
data[:, 1] = torch.rand(data_size)*300

#fill in risk data
data[:, 2] = torch.rand(data_size)*0.4

#fill in time to maturity data
data[:, 3] = torch.rand(data_size)*(150)+30

#fill in risk free rate data
data[:, 4] = torch.rand(data_size)*0.1

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x = self.data[index]
        y = black_scholes_call_price(x)
        y = y.squeeze()
        return torch.tensor(x, dtype=torch.float32), y