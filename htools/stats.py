import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns


def randomization_test(y1, y2, b=1_000, alternative='not_equal'):
    """Conduct a randomization test and return p-values for the difference in 
    means and the difference in standard deviations, respectively. (We don't
    require all permutations of the data to be taken so it's not technically a 
    permutation test.)
    
    y1 : array
        Array of floats or integers from sample 1.
    y2 : array
        Array of floats or integers from sample 2.
    b : int
        # of permutations
    alternative : str
        Alternative hypothesis to be tested. One of 
        ('less_than', 'not_equal', 'greater_than').
    """
    y_diff = y1.mean() - y2.mean()
    std_diff = y1.std() - y2.std()
    combined = np.concatenate((y1, y2))
    idx = len(y1)
    func = dict(less_than=lambda x, y: np.mean(np.array(x) < y),
                not_equal=lambda x, y: np.mean(np.abs(x) > np.abs(y)),
                greater_than=lambda x, y: np.mean(np.array(x) > y))
    means = []
    stds = []
    
    # Randomly assign each point to sample 1 or sample 2 and compute stats.
    for i in range(b):
        np.random.shuffle(combined)
        y1 = combined[:idx]
        y2 = combined[idx:]
        means.append(y1.mean() - y2.mean())
        stds.append(y1.std() - y2.std())
        
    # Calculate p-values.
    means, stds = np.array(means), np.array(stds)
    p_mean = func[alternative](means, y_diff)
    p_std = func[alternative](stds, std_diff)
    return p_mean, p_std


def plot_permutation_results(data, stat):
    """
    data : dict
        Dictionary keys should be integers specifying delta, while each value
        should be a list of p-values (floats) obtained from different 
        permutations.
    stat : str
        Specify what statistic the test is for. One of ('mean', 'std').
    """
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    plt.subplots_adjust(hspace=.4)
    for i, d in enumerate(delta):
        ax_i = ax[i//2][i%2]
        ax_i.plot(range(10, 1_010, 10), data[d], label=d)
        ax_i.legend(title='Delta', loc='upper left')
        ax_i.set_xlabel('B (# of Permutations)')
        ax_i.set_ylabel('P Value')
    fig.suptitle(f'Permutation Test of 2 {stat.title()}s')
    plt.show()


def min_sample_size(delta, sigma_sq, power, k, alternative='neq', alpha=.05):
    """Calculate minimum sample size for a 2 sample Z test.
    
    Parameters
    -----------
    delta : float
        Un-normalized effect size (mu1 - mu2).
    sigma_sq : float
        Population variance.
    power : float
        1 - Beta (desired value)
    k : float
        ratio of n1 to n2 (i.e. n_1=kn_2).
    alternative : str
        Alternative hypothesis, one of ('lt', 'neq', 'gt').
    alpha : float
        Significance level.
        
    Returns
    --------
    int
        Minimum sample size for desired power and effect size.
    """
    if alternative == 'neq':
        alpha /= 2
    z_crit = norm.ppf(alpha)
    z_pow = norm.ppf(power)
    n2 = (1 + 1/k) * (z_crit - z_pow)**2 * sigma_sq / delta**2
    n1 = k * n2
    return np.ceil(n1), np.ceil(n2)


def prop_min_sample_size(delta, pi1, pi2, power, k, alternative, alpha):
    """Calculate minimum sample size for a 2 proportion z test. 
    
    Note: Some power calculations for comparison of proportions will use a 
    pooled proportion from the 2 samples in the calculations. This variant 
    leaves pi1 and pi2 separate.
    
    Parameters
    -----------
    delta : float
        Un-normalized effect size (difference in proportions).
    pi1 : float
        Proportion for group 1.
    pi2 : float
        Proportion for group 2.
    power : float
        1 - Beta (desired value)
    k : float
        ratio of n1 to n2 (i.e. n_1=kn_2).
    alternative : str
        Alternative hypothesis, one of ('lt', 'neq', 'gt').
    alpha : float
        Significance level
        
    Returns
    --------
    int
        Minimum sample size for desired power and effect size.
    """
    if alternative == 'neq':
        alpha /= 2
    z_crit = norm.ppf(alpha)
    z_pow = norm.ppf(power)
    n2 = (z_crit - z_pow)**2 * (pi1 * (1-pi1) / k + pi2 * (1-pi2)) / delta**2
    n1 = k * n2
    return np.ceil(n1), np.ceil(n2)


def ztest_2prop(pi1, pi2, n2, k, d=0.0, alternative='neq'):
    """Conduct a 2 proportion z test. Alternate hypothesis can be read as
    {p1} {alternative} {p2}. The order of pi1 and pi2 is important if choosing
    a 1-sided alternative. This variation of the test does not use a pooled
    proportion to calculate the variance, as is sometimes done.
    
    Parameters
    -----------
    pi1 : float
        Sample proportion for group 1.
    pi2 : float
        Sample proportion for group 2.
    n2 : int
        Sample size of group 2.
    k : float
        Ratio of n1:n2 (i.e. n1 = k*n2).
    d : float
        Effect size, default 0.0. (d=0 means we are testing against the null
        hypothesis that the two proportions are equal. d>0 means we are 
        testing the null that p1 > p2.)
    alternative : str
        Alternative hypothesis, one of ('lt', 'neq', 'gt').
        Stands for 'less than', 'not equal', and 'greater than'.

    Returns
    --------
    float
        P-value.
    """
    n1 = k * n2
    se = np.sqrt(pi1*(1-pi1) / n1 + pi2*(1-pi2) / n2)
    z = (pi1 - pi2 - d) / se
    p_low = norm.cdf(z)
    p_high = 1 - norm.cdf(z) 
    if alternative == 'neq':
        return 2 * min(p_low, p_high)
    elif alternative == 'lt':
        return p_low
    elif alternative == 'gt':
        return p_high


def power_2sample_ztest(d, n2, k, sigma_sq, alternative='neq', alpha=.05):
    """Calculate the power of a 2 sample z test. We treat effect size as
    a positive quantity.
    
    Parameters
    -----------
    d : float
        Un-normalized effect size: np.abs(mu1 - mu2).
    n2 : int
        Sample size of group 2.
    k : float
        ratio of n1 to n2 (i.e. n1=k*n2).
    alternative : str
        Alternative hypothesis, one of ('lt', 'neq', 'gt').
        Stands for 'less than', 'not equal', 'greater than'.
    alpha : float
        Significance level
        
    Returns
    --------
    float
        Power of the hypothesis test.
    """
    if alternative == 'neq':
        alpha /= 2
    d = np.abs(d)
    n1 = k * n2
    z_crit = norm.ppf(1 - alpha)
    se = np.sqrt(sigma_sq * (1/n1 + 1/n2))
    return 1 - norm.cdf(z_crit - d / se)


def power_2prop_ztest(d, pi1, pi2, n2, k, alternative='neq', alpha=.05):
    """Calculate the power of a 2 proportion z test. This variant of the 
    calculation does not pool proportions between the two samples.
        
    Parameters
    -----------
    d : float
        Un-normalized effect size (pi1 - pi2).
    n2 : int
        Sample size of group 2.
    k : float
        ratio of n1 to n2 (i.e. n1=k*n2).
    alternative : str
        Alternative hypothesis, one of ('lt', 'neq', 'gt').
        Stands for 'less than', 'not equal', 'greater than'.
    alpha : float
        Significance level
        
    Returns
    --------
    float
        Power of the hypothesis test.
    """
    if alternative == 'neq':
        alpha /= 2
    d = np.abs(d)
    n1 = k * n2
    z_crit = norm.ppf(1 - alpha)
    se = np.sqrt((pi1*(1-pi1)) * 1/n1 + (pi2*(1-pi2)) * 1/n2)
    return 1 - norm.cdf(z_crit - d / se)


def bonferroni_correction(ps):
    """Apply Bonferroni correction to array of p values.
    
    Parameters
    -----------
    ps : array
        P-values for m hypothesis tests.
        
    Returns
    --------
    array
        Adjusted p-values (unsorted).
    """
    return np.minimum(ps * len(ps), 1)


def bh_correction(ps):
    """Apply Benjamini-Hochberg adjustment to array of p values.
    
    Parameters
    -----------
    ps : array
        P-values for m hypothesis tests.
        
    Returns
    --------
    array
        Adjusted p-values sorted in ascending order.
    """
    m = len(ps)
    p_new = np.sort(ps) * m / np.arange(1, m+1)
    p_adj = np.zeros_like(p_new)
    
    # Last value is not adjusted.
    p_adj[-1] = p_new[-1]
    for i in range(-2, -m-1, -1):
        p_adj[i] = min(p_new[i], p_adj[i+1])
    return p_adj


def bh_correct_power(d, n2, k, sig_sq, num_tests, alternative='neq', alpha=.05):
    """Calculate power values for a series of 2 sample z tests (consisting of
    {num_tests} total tests).
    """
    adj_alphas = np.arange(1, num_tests+1) * alpha / num_tests
    powers = []
    for alpha in adj_alphas:
        powers.append(power_2sample_ztest(d, n2, k, sig_sq, alternative, 
                                          alpha))
    return powers
