import numpy as np
from scipy.stats import norm

print(1.-0.05-2*(1-norm.cdf(0.56/(4./np.sqrt(28)))))