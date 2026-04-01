import asyncio
import json
import os
import threading
import time
import pytest

from nanobot.agent.resource_manager import WeightedSemaphore
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from unittest.mock import AsyncMock, MagicMock

# Alias for the global semaphore (simulate FEATHER_LIMIT)
FEATHER_LIMIT = WeightedSemaphore(4)

def test_manager_blocks_worker():
    FEATHER_LIMIT.current_usage = 0  # Reset for test
    manager_acquired = threading.Event()
    worker_started = threading.Event()
    worker_acquired = threading.Event()

    def manager_task():
        FEATHER_LIMIT.acquire(4)
        manager_acquired.set()
        time.sleep(0.3)
        FEATHER_LIMIT.release(4)

    def worker_task():
        worker_started.set()
        FEATHER_LIMIT.acquire(2)
        worker_acquired.set()
        FEATHER_LIMIT.release(2)

    t1 = threading.Thread(target=manager_task)
    t2 = threading.Thread(target=worker_task)
    t1.start()
    manager_acquired.wait(timeout=1)
    assert manager_acquired.is_set()
    t2.start()
    worker_started.wait(timeout=1)
    assert worker_started.is_set()

    # Worker should not acquire until manager releases
    assert not worker_acquired.is_set()

    t1.join(timeout=1)
    assert not t1.is_alive()
    t2.join(timeout=1)
    assert worker_acquired.is_set()

def test_interruptible_wait():
    FEATHER_LIMIT.current_usage = 0  # Reset for test
    interrupt_event = threading.Event()
    acquired = threading.Event()
    released = threading.Event()

    def blocking_task():
        FEATHER_LIMIT.acquire(4)
        acquired.set()
        # Wait for interrupt
        while not interrupt_event.is_set():
            time.sleep(0.05)
        FEATHER_LIMIT.release(4)
        released.set()

    t = threading.Thread(target=blocking_task)
    t.start()
    assert acquired.wait(timeout=1), "blocking_task did not acquire in time"

    # Try to acquire in another thread (should block)
    def waiter():
        FEATHER_LIMIT.acquire(2)
        FEATHER_LIMIT.release(2)
    w = threading.Thread(target=waiter)
    w.start()

    time.sleep(0.1)
    assert not released.is_set()

    interrupt_event.set()
    t.join(timeout=1)
    assert not t.is_alive(), "blocking_task did not terminate in time"
    w.join(timeout=1)
    assert not w.is_alive(), "waiter did not terminate in time"
    assert released.is_set()


@pytest.mark.featherless
def test_calculate_subagent_concurrency_points_default_map():
    # Featherless-like model naming should map to configured points
    assert SubagentManager._calculate_concurrency_points("featherless/72B") == 4
    assert SubagentManager._calculate_concurrency_points("featherless/32B") == 2
    assert SubagentManager._calculate_concurrency_points("featherless/24B") == 1


@pytest.mark.featherless
def test_calculate_subagent_concurrency_points_custom_map():
    assert SubagentManager._calculate_concurrency_points("featherless/80B", {"80B": 8}) == 8


# We only support featherless concurrency settings for these tests.
# Non-featherless models are out of the current supported path.

@pytest.mark.featherless
@pytest.mark.skipif(
    not os.getenv("TEST_FEATHERLESS_SUBSCRIPTION")
    or not os.getenv("FEATHERLESS_API_KEY"),
    reason="Featherless subscription tests are disabled unless TEST_FEATHERLESS_SUBSCRIPTION and FEATHERLESS_API_KEY are set",
)
@pytest.mark.asyncio
async def test_subagent_spawn_respects_concurrency_map(tmp_path, monkeypatch):
    tier = os.getenv("FEATHERLESS_SUBSCRIPTION_TIER_OVERRIDE", "72B")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "concurrencyMap": {
                        "72B": 2,
                        "32B": 1,
                        "4B": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: config_path)

    local_limit = WeightedSemaphore(4)
    monkeypatch.setattr("nanobot.agent.subagent.FEATHER_LIMIT", local_limit)

    provider = MagicMock()
    provider.get_default_model.return_value = f"featherless/{tier}"

    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    async def fake_run_subagent(*args, **kwargs):
        await asyncio.sleep(0)

    monkeypatch.setattr(manager, "_run_subagent", fake_run_subagent)

    assert local_limit.current_usage == 0
    result = await manager.spawn(task="check", label="featherless-test")
    assert "Subagent" in result
    await asyncio.sleep(0.05)
    assert local_limit.current_usage == 0


@pytest.mark.asyncio
async def test_subagent_spawn_times_out_when_concurrency_exhausted(tmp_path, monkeypatch):
    local_limit = WeightedSemaphore(2)
    local_limit.acquire(2)  # occupy all points
    monkeypatch.setattr("nanobot.agent.subagent.FEATHER_LIMIT", local_limit)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "concurrencyMap": {
                        "72B": 4,
                        "32B": 2,
                        "4B": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: config_path)

    provider = MagicMock()
    provider.get_default_model.return_value = "featherless/32B"

    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())
    manager._run_subagent = AsyncMock()

    result = await manager.spawn(task="check", label="featherless-timeout", acquire_timeout=0.1)
    assert "timed out" in result

    local_limit.release(2)

