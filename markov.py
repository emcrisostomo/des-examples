import numpy as np
import time

def simulate_markov_chain(lam=1/3, mttr_days=1, max_time=200_000):
    mu = 365 / mttr_days  # repair rate per year
    state = 0
    t = 0.0
    times = [t]
    states = [state]
    rng = np.random.default_rng(int(time.time()))
    while state < 3 and t < max_time:
        # Possible transitions and their rates
        transitions = []
        rates = []
        # Failure transitions
        if state == 0:
            transitions.append(1)
            rates.append(3 * lam)
        elif state == 1:
            transitions.extend([0, 2])
            rates.extend([mu, 2 * lam])
        elif state == 2:
            transitions.extend([1, 3])
            rates.extend([mu, 1 * lam])
        # No transitions from state 3 (absorbing)
        total_rate = sum(rates)
        if total_rate == 0:
            break
        dt = rng.exponential(1 / total_rate)
        t += dt
        # Choose which transition occurs
        probs = np.array(rates) / total_rate
        next_state = rng.choice(transitions, p=probs)
        state = next_state
        times.append(t)
        states.append(state)
    return times, states

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    times, states = simulate_markov_chain()
    plt.step(times, states, where='post')
    plt.xlabel('Time (years)')
    plt.ylabel('State')
    plt.title('3-Disk Mirror Markov Chain Simulation (with repair)')
    plt.yticks([0,1,2,3], ['All healthy', '1 failed', '2 failed', 'All failed'])
    plt.show()