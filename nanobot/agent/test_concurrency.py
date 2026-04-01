import threading
import time
import pytest
from .resource_manager import WeightedSemaphore

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
    worker_started.wait()
    time.sleep(0.05)
    t2.start()
    manager_acquired.wait()
    # Worker should not acquire until manager releases
    assert not worker_acquired.is_set()
    t1.join()
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
    acquired.wait()
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
    w.join(timeout=1)
    assert released.is_set()
