import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats, optimize

plt.rcParams['figure.dpi'] = 200


def metropolis_chain(starting_point, size, prior_logpdf, likelihood_logpdf, random_state):
    current_point = starting_point
    samples = np.empty((size, np.squeeze(starting_point).shape or 1))
    samples[0] = starting_point

    def density_log(alpha):
        prior_lp = prior_logpdf(alpha)
        likelihood_lp = likelihood_logpdf(alpha)
        return prior_lp + likelihood_lp

    for i in range(size):
        proposed_point = random_state.normal(current_point, 50)

        previous_lp, proposed_lp = density_log([current_point, proposed_point])
        jump_prob = np.exp(proposed_lp - previous_lp)

        if jump_prob > random_state.random():
            current_point = proposed_point

        samples[i] = current_point

    return samples


def plot_chains(chains, title, ax: plt.Axes):
    ax.set_title(title)
    for chain in chains:
        ax.plot(chain, alpha=0.5)


def get_posterior(data, prior, title, random_state):
    chain_size = 2000
    starting_points = (1e-5, 500, 750, 1000)
    chains = []
    for starting_point in starting_points:
        chains.append(metropolis_chain(
            starting_point=starting_point,
            size=chain_size,
            prior_logpdf=prior.logpdf,
            likelihood_logpdf=stats.gaussian_kde(data).logpdf,
            random_state=random_state
        ))

    chains = np.hstack(chains).T
    warm_up = 400
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    fig.suptitle(title)
    axes[0].set_title("Chains convergence during MCMC warmup")
    for i, chain in enumerate(chains, start=1):
        axes[0].plot(chain[:warm_up], alpha=0.5, label=f'Chain #{i}')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("LOS")

    chains_main = chains[:, warm_up:]
    rng = (0, 1200)
    grid = np.linspace(*rng, 1000)
    axes[1].set_title("Histogram of the posterior")
    axes[1].plot(grid, prior.pdf(grid), '--k', alpha=0.75)
    axes[1].hist(chains_main.ravel(), range=rng, bins=100, color='b', alpha=0.7, density=True)
    axes[1].plot(data.where(data < 1200), np.full(len(data), 0.0001), '|', color='k')
    axes[1].set_xlabel("LOS")
    axes[1].set_ylabel("Density")
    axes[1].legend(["Prior", "Data points", "Posterior"])

    plt.show()
    return chains_main


def get_data():
    data = pd.read_csv('data/Dataset.csv', sep=',', header=0, index_col=0)
    data = data[['Assignment', 'LOS']]
    display(data.head(7).style.set_caption('Example of the data'))
    data = data.groupby('Assignment')
    display(data.describe().style.set_caption('LOS statistics for each treatment').format('{:.2f}'))

    fig, axes = plt.subplots(1, len(data), figsize=(15, 5), sharex='all', sharey='all',
                             layout='constrained')
    fig.suptitle('LOS distribution in different treatment groups')
    for i, (title, df) in enumerate([
        ("Standard", data.get_group('S')),
        ("Guided", data.get_group('G'))
    ]):
        ax = axes[i]
        ax.set_title(title)
        ax.hist(df['LOS'], bins=100, range=(0, 1200))
        ax.set_xlabel('LOS')
        ax.set_ylabel('Count')
    plt.show()
    return data.get_group('S'), data.get_group('G')


def get_prior() -> stats.rv_continuous:
    prior: stats.rv_continuous = stats.gengamma(a=1.05, c=1, scale=1000)
    m, v = prior.stats(moments='mv')
    low, high = prior.ppf([0.025, 0.975])
    mode = optimize.minimize(
        lambda x: -prior.logpdf(x), x0=m, bounds=[(0, 10000)], method='Nelder-Mead').x[0]
    description = pd.DataFrame({
        'Mean': [m],
        'Variance': [v],
        'Mode': [mode],
        '95% CI': [f'[{low:.2f}, {high:.2f}]']
    })
    display(description.style.set_caption('Prior properties').format(
        lambda x: x if isinstance(x, str) else f'{x:.2f}'))
    return prior


def compare_posteriors(posterior_s, posterior_g):
    difference = posterior_s.ravel() - posterior_g.ravel()
    ax = arviz.plot_posterior(
        difference, kind='kde', figsize=(10, 5), hdi_prob=0.95,
        backend_kwargs={'layout': 'constrained'})
    ax.set_title('Distribution of the difference between LOS for standard and guided treatment')
    plt.show()


def main():
    rs = np.random.RandomState(1337)
    prior = get_prior()
    standard_treatment, guided_treatment = get_data()

    standard_samples = get_posterior(standard_treatment['LOS'], prior, 'Standard treatment', rs)
    guided_samples = get_posterior(guided_treatment['LOS'], prior, 'Guided treatment', rs)

    compare_posteriors(standard_samples, guided_samples)


if __name__ == "__main__":
    main()
