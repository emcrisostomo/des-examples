import numpy as np
import time
from multiprocessing import Pool, cpu_count

# 3-disk RAID system Markov chain simulation
def simulate_markov_chain(lam=1/3, mttr_days=1, max_time=2_000, seed=None):
    mu = 365 / mttr_days  # repair rate per year
    # Memoize transitions and rates for each state
    transitions_dict = {
        0: [1],
        1: [0, 2],
        2: [1, 3]
    }
    rates_dict = {
        0: [3 * lam],
        1: [mu, 2 * lam],
        2: [mu, 1 * lam]
    }
    state = 0
    t = 0.0
    times = [t]
    states = [state]
    rng = np.random.default_rng(seed)
    while state < 3 and t < max_time:
        transitions = transitions_dict.get(state, [])
        rates = rates_dict.get(state, [])
        total_rate = sum(rates)
        if total_rate == 0:
            break
        # In a continuous-time Markov chain, each possible transition {X_i} is 
        # modeled as an independent exponential clock with rate equal to the
        # transition rate.
        # The time to the next transition is min{X_i} exponentially distributed
        # with rate equal to the sum of the rates {X_i} of all possible
        # transitions.
        dt = rng.exponential(1 / total_rate)
        t += dt
        # The index of the next state is chosen with probability proportional
        # to the rates of the possible transitions:
        # P(next state = j) = rate(j) / sum(rate(i))
        probs = np.array(rates) / total_rate
        next_state = rng.choice(transitions, p=probs)
        state = next_state
        times.append(t)
        states.append(state)
    return times, states

if __name__ == "__main__":
    num_simulations = 100  # Increase for better estimate
    seeds = [int(time.time()) + i for i in range(num_simulations)]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            simulate_markov_chain,
            [(1/3, 1, 200_000, seed) for seed in seeds]
        )

    # Collect the time to absorption (state 3) for each simulation
    failure_times = []
    for times, states in results:
        if states[-1] == 3:
            failure_times.append(times[-1])
        else:
            failure_times.append(np.nan)

    # Compute mean, ignoring runs that did not fail
    mttf = np.nanmean(failure_times)
    print(f"Estimated Mean Time To Failure (MTTF): {mttf:.2f} years")

    # Optional: plot histogram of failure times
    import matplotlib.pyplot as plt
    plt.hist([t for t in failure_times if not np.isnan(t)], bins=20, color='skyblue')
    plt.xlabel('Time to Failure (years)')
    plt.ylabel('Count')
    plt.title('Distribution of Time to Failure')
    plt.show()