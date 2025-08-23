# pip install simpy numpy
import math, random, statistics
import numpy as np
import simpy

def rates_from_A_MTTR(A, MTTR_hours):
    mu = 1.0 / MTTR_hours
    lam = mu * (1 - A) / A
    return lam, mu

class Component:
    def __init__(self, env, name, ttf_sample, ttr_sample, on_state_change):
        self.env = env
        self.name = name
        self.ttf_sample = ttf_sample
        self.ttr_sample = ttr_sample
        self.on_state_change = on_state_change
        self.up = True
        self.proc = env.process(self._run())

    def _run(self):
        while True:
            self.up = True
            self.on_state_change(self.name, True, self.env.now)
            try:
                # Uptime until failure
                dt = max(0.0, self.ttf_sample())
                yield self.env.timeout(dt)
                # Fail
                self.up = False
                self.on_state_change(self.name, False, self.env.now)
                # Repair time
                dt = max(0.0, self.ttr_sample())
                yield self.env.timeout(dt)
            except simpy.Interrupt as intr:
                # Forced failure (e.g., common-cause)
                if self.up:
                    self.up = False
                    self.on_state_change(self.name, False, self.env.now)
                    dt = max(0.0, self.ttr_sample())
                    yield self.env.timeout(dt)
                # else already down; ignore

    def force_fail_now(self):
        # Interrupt the process so it enters the repair branch immediately
        if self.proc.is_alive:
            self.proc.interrupt('force_fail')

class SystemAvailability:
    def __init__(self, env, need_k_of_n, component_names):
        self.env = env
        self.need_k = need_k_of_n
        self.up_components = set(component_names)
        self.down_start = None
        self.outages = []
        self.total_downtime = 0.0

    def on_state_change(self, name, is_up, t):
        if is_up:
            self.up_components.add(name)
        else:
            self.up_components.discard(name)

        is_system_up = (len(self.up_components) >= self.need_k)
        if is_system_up and self.down_start is not None:
            # System recovers
            dt = t - self.down_start
            self.total_downtime += dt
            self.outages.append(dt)
            self.down_start = None
        elif (not is_system_up) and self.down_start is None:
            # System goes down
            self.down_start = t

    def finalize(self, t_end):
        if self.down_start is not None:
            dt = t_end - self.down_start
            self.total_downtime += dt
            self.outages.append(dt)
            self.down_start = None

def exp_sampler(rate_per_hour):
    # Exponential TTF
    return lambda: random.expovariate(rate_per_hour)

def lognormal_sampler(median_hours, gsd):
    # Lognormal parameterized by median and geometric std dev
    # median = exp(mu), gsd = exp(sigma)
    mu = math.log(median_hours)
    sigma = math.log(gsd)
    return lambda: np.random.lognormal(mean=mu, sigma=sigma)

def run_one(seed, horizon_hours=24*365,
            A=0.999, MTTR_hours=1.0,
            A2=None, MTTR2_hours=None,
            common_cause_rate=0.0):
    random.seed(seed); np.random.seed(seed)
    env = simpy.Environment()

    # Derive rates
    lam1, _ = rates_from_A_MTTR(A, MTTR_hours)
    lam2, _ = rates_from_A_MTTR(A2 if A2 else A, MTTR2_hours if MTTR2_hours else MTTR_hours)

    # Samplers
    ttf1 = exp_sampler(lam1)
    ttf2 = exp_sampler(lam2)
    # Repairs as lognormal: choose median=MTTR, gsd=2.0 (tune to ops data)
    ttr1 = lognormal_sampler(MTTR_hours, gsd=2.0)
    ttr2 = lognormal_sampler(MTTR2_hours if MTTR2_hours else MTTR_hours, gsd=2.0)

    system = SystemAvailability(env, need_k_of_n=1, component_names=['A','B'])

    c1 = Component(env, 'A', ttf1, ttr1, system.on_state_change)
    c2 = Component(env, 'B', ttf2, ttr2, system.on_state_change)

    def common_cause():
        if common_cause_rate <= 0: return
        while True:
            dt = random.expovariate(common_cause_rate)
            yield env.timeout(dt)
            # Knock both down immediately
            c1.force_fail_now()
            c2.force_fail_now()

    env.process(common_cause())
    env.run(until=horizon_hours)
    system.finalize(horizon_hours)

    total = horizon_hours
    availability = 1.0 - system.total_downtime / total
    return {
        "availability": availability,
        "total_downtime_hours": system.total_downtime,
        "outage_count": len(system.outages),
        "p95_outage_minutes": (np.percentile(system.outages, 95)*60.0) if system.outages else 0.0,
        "monthly_downtime_minutes": system.total_downtime * 60.0 / (total/ (24*30.0))
    }

def experiment(replications=200, **kwargs):
    results = [run_one(seed=10_000+i, **kwargs) for i in range(replications)]
    mdm = [r["monthly_downtime_minutes"] for r in results]
    return {
        "mean_monthly_minutes": statistics.mean(mdm),
        "p50_monthly_minutes": statistics.median(mdm),
        "p95_monthly_minutes": np.percentile(mdm, 95),
        "mean_availability": statistics.mean(r["availability"] for r in results),
        "mean_outage_count": statistics.mean(r["outage_count"] for r in results),
        "p95_outage_minutes": np.percentile([r["p95_outage_minutes"] for r in results], 95)
    }

# Example usage:
# Two nodes with A=99.9%, MTTR=1h, plus rare common-cause at 1 per year:
summary = experiment(replications=500, A=0.999, MTTR_hours=1.0, common_cause_rate=1/ (24*365))
print(summary)
