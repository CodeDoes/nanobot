#!/usr/bin/env python3
"""Quick local demo: visualize featherless-style concurrency point scheduling.

Usage:
    python tools/concurrency_demo.py --workers 5 --slots 4
    python tools/concurrency_demo.py --tasks 72B,32B,32B,4B --slots 4

This script is designed for docs/demo and prints both event log and tmux-like timeline.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from typing import List

from nanobot.agent.resource_manager import WeightedSemaphore

MODEL_SIZE_POINTS = {
    "72B": 4,
    "32B": 2,
    "4B": 1,
}


def model_points(model_id: str, custom_map: dict[str, int] | None = None) -> int:
    if custom_map is None:
        custom_map = MODEL_SIZE_POINTS
    # extract token portion, e.g. featherless/72B -> 72B
    size = model_id.split("/")[-1]
    return custom_map.get(size, 1)


def run_demo(tasks: List[str], slots: int, tick: float = 0.2, seed: int | None = None, temp: float = 1.0) -> None:
    if seed is not None:
        random.seed(seed)

    semaphore = WeightedSemaphore(slots)
    timeline_lock = threading.Lock()

    # timeline per task: list of states per tick
    timelines: list[list[str]] = [[] for _ in tasks]
    events: list[str] = []

    start_time = time.time()
    active = True

    def worker(idx: int, model_id: str):
        points = model_points(model_id)
        events.append(f"[{idx}] wants {model_id} ({points} points)")

        # wait until acquired
        while True:
            with timeline_lock:
                timelines[idx].append("W")
            if semaphore.current_usage + points <= semaphore.max_points:
                break
            time.sleep(tick)

        semaphore.acquire(points)
        events.append(f"[{idx}] acquired {points} points")
        with timeline_lock:
            timelines[idx].append("A")

        # simulate actual work; deterministic with seed/temp
        run_ticks = max(1, int(round((1 + temp) * 2)))
        for _ in range(run_ticks):
            with timeline_lock:
                timelines[idx].append("X")
            time.sleep(tick)

        semaphore.release(points)
        events.append(f"[{idx}] released {points} points")

        # tail idle after done
        for _ in range(2):
            with timeline_lock:
                timelines[idx].append(".")
            time.sleep(tick)

    threads = []
    for i, model in enumerate(tasks):
        t = threading.Thread(target=worker, args=(i, model), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print("\n=== Event Log ===")
    for e in events:
        print(e)

    print("\n=== Timeline (tmux style) ===")
    print("slot   " + "".join([f"{i:>2}" for i in range(len(timelines[0]))]))
    for idx, line in enumerate(timelines):
        row = "".join(line)
        print(f"{idx:>2}:    {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Featherless concurrency timeline demo")
    parser.add_argument("--slots", type=int, default=4, help="Total concurrency points")
    parser.add_argument("--tasks", type=str, default="72B,32B,32B,4B", help="Comma-separated model sizes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic behavior")
    parser.add_argument("--temp", type=float, default=0.0, help="Simulation 'temperature' to vary run length (0 stable)")
    parser.add_argument("--tick", type=float, default=0.2, help="Tick duration in seconds")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_demo(tasks=tasks, slots=args.slots, tick=args.tick, seed=args.seed, temp=args.temp)


if __name__ == "__main__":
    main()
