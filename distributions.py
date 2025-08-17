import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, gamma


def demo_poisson():
    fig, ax = plt.subplots(1, 1)
    mu = 5
    f = poisson(mu)

    # see mean equal to variance equal to mu!
    mean, var, skew, kurt = f.stats(moments='mvsk')
    print(f"mean: {mean}, var: {var}, skew: {skew}, kurt: {kurt}")
    x = np.arange(f.ppf(0.01), f.ppf(0.99))

    # PDF is not defined, since it's not a continuous function!
    ax.plot(x, f.pmf(x), 'bo', ms=8, label='poisson pmf')

    ax.vlines(x, 0, f.pmf(x), colors='b', lw=5, alpha=0.5)


def demo_gamma():
    f = gamma(2, 0, 1)


if __name__ == '__main__':
    demo_poisson()
    plt.show()
