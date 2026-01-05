# aws_availability_simpy_idiomatic.py
# Discrete-Event Simulation for AWS-style availability decisions using SimPy.
# - Three scenarios: SR-MAZ, SR-MAZ + SQS buffer, MR-AA.
#
# Usage:
#   python aws_availability_simpy_idiomatic.py             # run defaults
#   python aws_availability_simpy_idiomatic.py --help      # see knobs
#
# Outputs:
#   - aws_availability_summary.csv  (one-row-per-scenario aggregate metrics)

import math
import statistics
import csv
import argparse
import concurrent.futures
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple

import numpy as np
import simpy

# ---------- Helpers ----------


def rates_from_a_and_mtrr(a: float, mttr_hours: float) -> float:
    """Return failure rate lambda (per hour) implied by steady-state A and mean MTTR."""
    mu = 1.0 / mttr_hours
    lam = mu * (1.0 - a) / a
    return lam


def percentile(xs: List[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of floats.

    Returns:
        The value at the specified percentile, or 0.0 if the list is empty.
    """
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


# ---------- SimPy Model ----------


@dataclass
class ComponentSpec:
    name: str
    lam_per_h: float
    ttr_median_h: float
    ttr_gsd: float
    blip_seconds: float = 0.0


@dataclass
class CCGroup:
    name: str
    rate_per_h: float
    members: List[str]
    blip_seconds: float = 0.0


@dataclass
class Metrics:
    system_down_time: float
    blip_hours: float
    system_outages: List[float]


class Model:
    def __init__(
        self,
        env: simpy.Environment,
        horizon_h: float,
        components: Dict[str, ComponentSpec],
        groups: List[CCGroup],
        is_system_up: Callable[[Dict[str, bool]], bool],
        rng: np.random.Generator,
    ):
        self.env = env
        self.horizon_h = horizon_h
        self.components = components
        self.groups = groups
        self.is_system_up = is_system_up
        self.rng = rng

        self.state = dict.fromkeys(components.keys(), True)
        self.force_fail_events: Dict[str, simpy.Event] = {
            name: simpy.Event(env) for name in components.keys()
        }

        self.was_up = is_system_up(self.state)
        self.system_down_start = None
        self.system_outages: List[float] = []
        self.system_down_time = 0.0
        self.blip_hours = 0.0

    def add_blip(self, blip_seconds: float) -> None:
        if blip_seconds > 0:
            self.blip_hours += blip_seconds / 3600.0

    def on_state_change(self) -> None:
        now_up = self.is_system_up(self.state)
        if self.was_up and not now_up:
            self.system_down_start = self.env.now
        elif (not self.was_up) and now_up:
            dt = self.env.now - self.system_down_start
            if dt > 0:
                self.system_outages.append(dt)
                self.system_down_time += dt
            self.system_down_start = None
        self.was_up = now_up

    def finalize(self) -> Metrics:
        if not self.was_up and self.system_down_start is not None:
            dt = self.horizon_h - self.system_down_start
            if dt > 0:
                self.system_outages.append(dt)
                self.system_down_time += dt
        return Metrics(self.system_down_time, self.blip_hours, self.system_outages)


def sample_ttf(rng: np.random.Generator, lam_per_h: float) -> float:
    return rng.exponential(1 / lam_per_h)


def sample_ttr(rng: np.random.Generator, median_h: float, gsd: float) -> float:
    sigma = math.log(gsd)
    mu = math.log(median_h)
    return rng.lognormal(mean=mu, sigma=sigma)


def component_process(model: Model, spec: ComponentSpec):
    name = spec.name
    while True:
        ttf = sample_ttf(model.rng, spec.lam_per_h)
        failure_timeout = model.env.timeout(ttf)
        force_event = model.force_fail_events[name]
        result = yield failure_timeout | force_event
        forced = force_event in result

        if model.state[name]:
            model.state[name] = False
            if not forced:
                model.add_blip(spec.blip_seconds)
            model.on_state_change()

        ttr = sample_ttr(model.rng, spec.ttr_median_h, spec.ttr_gsd)
        yield model.env.timeout(ttr)

        if not model.state[name]:
            model.state[name] = True
            model.on_state_change()


def group_process(model: Model, group: CCGroup):
    while True:
        dt = model.rng.exponential(1 / group.rate_per_h)
        yield model.env.timeout(dt)
        model.add_blip(group.blip_seconds)
        for name in group.members:
            if model.state[name]:
                event = model.force_fail_events[name]
                if not event.triggered:
                    event.succeed()
                model.force_fail_events[name] = simpy.Event(model.env)


def build_model(
    horizon_h: float,
    components: Dict[str, ComponentSpec],
    groups: List[CCGroup],
    is_system_up: Callable[[Dict[str, bool]], bool],
    seed: int,
) -> Tuple[simpy.Environment, Model]:
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    model = Model(env, horizon_h, components, groups, is_system_up, rng)

    for spec in components.values():
        env.process(component_process(model, spec))

    for group in groups:
        if group.rate_per_h > 0:
            env.process(group_process(model, group))

    return env, model


def run_sim(
    horizon_h: float,
    components: Dict[str, ComponentSpec],
    groups: List[CCGroup],
    is_system_up: Callable[[Dict[str, bool]], bool],
    seed: int,
) -> Dict[str, float]:
    env, model = build_model(horizon_h, components, groups, is_system_up, seed)
    env.run(until=horizon_h)
    metrics = model.finalize()
    total_down = metrics.system_down_time + metrics.blip_hours
    availability = 1.0 - (total_down / horizon_h)
    return {
        "availability": availability,
        "interval_down_hours": metrics.system_down_time,
        "blip_hours": metrics.blip_hours,
        "total_down_hours": total_down,
        "outage_count": len(metrics.system_outages),
        "p95_outage_minutes": (
            percentile(metrics.system_outages, 95) * 60.0
            if metrics.system_outages
            else 0.0
        ),
    }


# ---------- Scenarios ----------


def scenario_sr_maz(
    horizon_days=365,
    replications=200,
    A_alb=0.9999,
    alb_MTTR_h=0.5,
    alb_blip_s=15,
    A_app=0.999,
    app_MTTR_h=0.5,
    app_blip_s=30,
    A_db=0.9995,
    db_MTTR_h=1.0,
    db_blip_s=120,
    az_event_per_year=0.5,
    region_event_per_year=0.1,
    region_blip_s=60,
    ttr_gsd=2.0,
    seed0=10000,
):
    """
    System up iff: ALB && (APP_AZ1 || APP_AZ2) && DB
    """
    horizon_h = horizon_days * 24.0
    az_rate = az_event_per_year / (365 * 24.0)
    region_rate = region_event_per_year / (365 * 24.0)

    lam_alb = rates_from_a_and_mtrr(A_alb, alb_MTTR_h)
    lam_app = rates_from_a_and_mtrr(A_app, app_MTTR_h)
    lam_db = rates_from_a_and_mtrr(A_db, db_MTTR_h)

    components = {
        "alb": ComponentSpec("alb", lam_alb, alb_MTTR_h, ttr_gsd, alb_blip_s),
        "app_az1": ComponentSpec("app_az1", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "app_az2": ComponentSpec("app_az2", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "db": ComponentSpec("db", lam_db, db_MTTR_h, ttr_gsd, db_blip_s),
    }
    groups = [
        CCGroup("az1", az_rate, ["app_az1"]),
        CCGroup("az2", az_rate, ["app_az2"]),
        CCGroup(
            "region", region_rate, list(components.keys()), blip_seconds=region_blip_s
        ),
    ]

    return run_reps(
        horizon_h, components, groups, is_up_sr_maz, replications, seed0, "SR-MAZ"
    )


def scenario_sr_maz_buffered(
    horizon_days=365,
    replications=200,
    A_alb=0.9999,
    alb_MTTR_h=0.5,
    alb_blip_s=15,
    A_app=0.999,
    app_MTTR_h=0.5,
    app_blip_s=30,
    A_sqs=0.9999,
    sqs_MTTR_h=0.25,
    sqs_blip_s=5,
    az_event_per_year=0.5,
    region_event_per_year=0.1,
    region_blip_s=60,
    ttr_gsd=2.0,
    seed0=20000,
):
    """
    System up iff: ALB && (APP_AZ1 || APP_AZ2) && SQS
    (DB removed from gating; writes are queued)
    """
    horizon_h = horizon_days * 24.0
    az_rate = az_event_per_year / (365 * 24.0)
    region_rate = region_event_per_year / (365 * 24.0)

    lam_alb = rates_from_a_and_mtrr(A_alb, alb_MTTR_h)
    lam_app = rates_from_a_and_mtrr(A_app, app_MTTR_h)
    lam_sqs = rates_from_a_and_mtrr(A_sqs, sqs_MTTR_h)

    components = {
        "alb": ComponentSpec("alb", lam_alb, alb_MTTR_h, ttr_gsd, alb_blip_s),
        "app_az1": ComponentSpec("app_az1", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "app_az2": ComponentSpec("app_az2", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "sqs": ComponentSpec("sqs", lam_sqs, sqs_MTTR_h, ttr_gsd, sqs_blip_s),
    }
    groups = [
        CCGroup("az1", az_rate, ["app_az1"]),
        CCGroup("az2", az_rate, ["app_az2"]),
        CCGroup(
            "region", region_rate, list(components.keys()), blip_seconds=region_blip_s
        ),
    ]

    return run_reps(
        horizon_h,
        components,
        groups,
        is_up_sr_maz_buffered,
        replications,
        seed0,
        "SR-MAZ + buffer(SQS)",
    )


def scenario_MR_AA(
    horizon_days=365,
    replications=200,
    A_alb=0.9999,
    alb_MTTR_h=0.5,
    alb_blip_s=15,
    A_app=0.999,
    app_MTTR_h=0.5,
    app_blip_s=30,
    A_db=0.9995,
    db_MTTR_h=1.0,
    db_blip_s=120,
    az_event_per_year=0.5,
    region1_event_per_year=0.08,
    region2_event_per_year=0.08,
    region_blip_s=120,
    ttr_gsd=2.0,
    seed0=30000,
):
    """
    Two Regions active-active.
    Up iff (ALB1 && (APP1_AZ1 || APP1_AZ2) && DB1) OR (ALB2 && (APP2_AZ1 || APP2_AZ2) && DB2)
    Region-level CC adds a larger blip to mimic cross-Region traffic shifting.
    """
    horizon_h = horizon_days * 24.0
    az_rate = az_event_per_year / (365 * 24.0)
    r1_rate = region1_event_per_year / (365 * 24.0)
    r2_rate = region2_event_per_year / (365 * 24.0)

    lam_alb = rates_from_a_and_mtrr(A_alb, alb_MTTR_h)
    lam_app = rates_from_a_and_mtrr(A_app, app_MTTR_h)
    lam_db = rates_from_a_and_mtrr(A_db, db_MTTR_h)

    components = {
        # Region 1
        "alb_r1": ComponentSpec("alb_r1", lam_alb, alb_MTTR_h, ttr_gsd, alb_blip_s),
        "app1_az1": ComponentSpec("app1_az1", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "app1_az2": ComponentSpec("app1_az2", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "db_r1": ComponentSpec("db_r1", lam_db, db_MTTR_h, ttr_gsd, db_blip_s),
        # Region 2
        "alb_r2": ComponentSpec("alb_r2", lam_alb, alb_MTTR_h, ttr_gsd, alb_blip_s),
        "app2_az1": ComponentSpec("app2_az1", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "app2_az2": ComponentSpec("app2_az2", lam_app, app_MTTR_h, ttr_gsd, app_blip_s),
        "db_r2": ComponentSpec("db_r2", lam_db, db_MTTR_h, ttr_gsd, db_blip_s),
    }
    groups = [
        # AZ CC
        CCGroup("r1_az1", az_rate, ["app1_az1"]),
        CCGroup("r1_az2", az_rate, ["app1_az2"]),
        CCGroup("r2_az1", az_rate, ["app2_az1"]),
        CCGroup("r2_az2", az_rate, ["app2_az2"]),
        # Region CC (bigger blip for cross-Region shift)
        CCGroup(
            "region1",
            r1_rate,
            ["alb_r1", "app1_az1", "app1_az2", "db_r1"],
            blip_seconds=region_blip_s,
        ),
        CCGroup(
            "region2",
            r2_rate,
            ["alb_r2", "app2_az1", "app2_az2", "db_r2"],
            blip_seconds=region_blip_s,
        ),
    ]

    return run_reps(
        horizon_h, components, groups, is_up_mr_aa, replications, seed0, "MR-AA"
    )


# ---------- Runner ----------


def single_run_chunk(
    start, count, horizon_h, components_args, groups_args, is_up_fn, seed0, label
):
    results = []
    months = horizon_h / (24.0 * 30.0)
    for i in range(start, start + count):
        components = {k: ComponentSpec(**v) for k, v in components_args.items()}
        groups = [CCGroup(**g) for g in groups_args]
        out = run_sim(horizon_h, components, groups, is_up_fn, seed=seed0 + i)
        monthly_min = out["total_down_hours"] * 60.0 / months
        results.append(
            {
                "scenario": label,
                "availability": out["availability"],
                "monthly_downtime_min": monthly_min,
                "outage_count": out["outage_count"],
                "p95_outage_minutes": out["p95_outage_minutes"],
            }
        )
    return results


def single_run_chunk_star(args):
    return single_run_chunk(*args)


def run_reps(horizon_h, components, groups, is_up_fn, replications, seed0, label):
    n_workers = os.cpu_count() or 2
    chunk_size = math.ceil(replications / n_workers)
    components_args = {k: vars(v) for k, v in components.items()}
    groups_args = [vars(g) for g in groups]
    args_list = []
    for w in range(n_workers):
        start = w * chunk_size
        count = min(chunk_size, replications - start)
        if count > 0:
            args_list.append(
                (
                    start,
                    count,
                    horizon_h,
                    components_args,
                    groups_args,
                    is_up_fn,
                    seed0,
                    label,
                )
            )

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunked_results = executor.map(single_run_chunk_star, args_list)

    results = [item for chunk in chunked_results for item in chunk]
    mdm = [r["monthly_downtime_min"] for r in results]
    avs = [r["availability"] for r in results]
    outs = [r["outage_count"] for r in results]
    p95o = [r["p95_outage_minutes"] for r in results]
    return {
        "scenario": label,
        "replications": replications,
        "mean_monthly_min": statistics.mean(mdm),
        "p50_monthly_min": statistics.median(mdm),
        "p95_monthly_min": percentile(mdm, 95),
        "mean_availability": statistics.mean(avs),
        "p50_availability": statistics.median(avs),
        "p95_availability": percentile(avs, 95),
        "mean_outage_count": statistics.mean(outs),
        "p50_outage_count": statistics.median(outs),
        "p95_outage_count": percentile(outs, 95),
        "p95_outage_minutes": percentile(p95o, 95),
    }, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replications", type=int, default=200)
    ap.add_argument("--horizon-days", type=float, default=365)
    ap.add_argument("--ttr-gsd", type=float, default=2.0)

    # quick knobs that matter most
    ap.add_argument("--az-cc-per-year", type=float, default=0.5)
    ap.add_argument("--region-cc-per-year", type=float, default=0.1)
    ap.add_argument("--region-blip-s", type=float, default=60)

    # component-level SLAs and MTTRs (edit to taste)
    ap.add_argument("--alb-A", type=float, default=0.9999)
    ap.add_argument("--alb-mttr-h", type=float, default=0.5)
    ap.add_argument("--alb-blip-s", type=float, default=15)

    ap.add_argument("--app-A", type=float, default=0.999)
    ap.add_argument("--app-mttr-h", type=float, default=0.5)
    ap.add_argument("--app-blip-s", type=float, default=30)

    ap.add_argument("--db-A", type=float, default=0.9995)
    ap.add_argument("--db-mttr-h", type=float, default=1.0)
    ap.add_argument("--db-blip-s", type=float, default=120)

    ap.add_argument("--sqs-A", type=float, default=0.9999)
    ap.add_argument("--sqs-mttr-h", type=float, default=0.25)
    ap.add_argument("--sqs-blip-s", type=float, default=5)

    args = ap.parse_args()

    # SR-MAZ
    sr_summary, _ = scenario_sr_maz(
        horizon_days=args.horizon_days,
        replications=args.replications,
        A_alb=args.alb_A,
        alb_MTTR_h=args.alb_mttr_h,
        alb_blip_s=args.alb_blip_s,
        A_app=args.app_A,
        app_MTTR_h=args.app_mttr_h,
        app_blip_s=args.app_blip_s,
        A_db=args.db_A,
        db_MTTR_h=args.db_mttr_h,
        db_blip_s=args.db_blip_s,
        az_event_per_year=args.az_cc_per_year,
        region_event_per_year=args.region_cc_per_year,
        region_blip_s=args.region_blip_s,
        ttr_gsd=args.ttr_gsd,
        seed0=int(time.time()),
    )

    # SR-MAZ + buffered writes (SQS)
    buf_summary, _ = scenario_sr_maz_buffered(
        horizon_days=args.horizon_days,
        replications=args.replications,
        A_alb=args.alb_A,
        alb_MTTR_h=args.alb_mttr_h,
        alb_blip_s=args.alb_blip_s,
        A_app=args.app_A,
        app_MTTR_h=args.app_mttr_h,
        app_blip_s=args.app_blip_s,
        A_sqs=args.sqs_A,
        sqs_MTTR_h=args.sqs_mttr_h,
        sqs_blip_s=args.sqs_blip_s,
        az_event_per_year=args.az_cc_per_year,
        region_event_per_year=args.region_cc_per_year,
        region_blip_s=args.region_blip_s,
        ttr_gsd=args.ttr_gsd,
        seed0=int(time.time()),
    )

    # MR-AA
    mr_summary, _ = scenario_MR_AA(
        horizon_days=args.horizon_days,
        replications=args.replications,
        A_alb=args.alb_A,
        alb_MTTR_h=args.alb_mttr_h,
        alb_blip_s=args.alb_blip_s,
        A_app=args.app_A,
        app_MTTR_h=args.app_mttr_h,
        app_blip_s=args.app_blip_s,
        A_db=args.db_A,
        db_MTTR_h=args.db_mttr_h,
        db_blip_s=args.db_blip_s,
        az_event_per_year=args.az_cc_per_year,
        region1_event_per_year=args.region_cc_per_year,
        region2_event_per_year=args.region_cc_per_year,
        region_blip_s=args.region_blip_s,
        ttr_gsd=args.ttr_gsd,
        seed0=int(time.time()),
    )

    rows = [sr_summary, buf_summary, mr_summary]
    with open("aws_availability_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # pretty print
    def pr(s):
        print(
            f"{s['scenario']}: "
            f"mean_availability={s['mean_availability']:.6f}, "
            f"mean_monthly_min={s['mean_monthly_min']:.2f}, "
            f"p95_monthly_min={s['p95_monthly_min']:.2f}, "
            f"p95_outage_minutes={s['p95_outage_minutes']:.1f}, "
            f"replications={s['replications']}"
        )

    pr(sr_summary)
    pr(buf_summary)
    pr(mr_summary)
    print("Wrote aws_availability_summary.csv")


# Move these to the top level:
def is_up_sr_maz(state: Dict[str, bool]) -> bool:
    return state["alb"] and (state["app_az1"] or state["app_az2"]) and state["db"]


def is_up_sr_maz_buffered(state: Dict[str, bool]) -> bool:
    return state["alb"] and (state["app_az1"] or state["app_az2"]) and state["sqs"]


def is_up_mr_aa(state: Dict[str, bool]) -> bool:
    def r1_up(s):
        return s["alb_r1"] and (s["app1_az1"] or s["app1_az2"]) and s["db_r1"]

    def r2_up(s):
        return s["alb_r2"] and (s["app2_az1"] or s["app2_az2"]) and s["db_r2"]

    return r1_up(state) or r2_up(state)


if __name__ == "__main__":
    main()
