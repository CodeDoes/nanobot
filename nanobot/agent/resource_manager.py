import threading
import time

class WeightedSemaphore:
    def __init__(self, max_points: int):
        self.max_points = max_points
        self.current_usage = 0
        self.condition = threading.Condition()

    def acquire(self, points: int, timeout: float | None = None) -> bool:
        with self.condition:
            if timeout is None:
                while self.current_usage + points > self.max_points:
                    self.condition.wait()
                self.current_usage += points
                return True

            end_time = time.monotonic() + timeout
            while self.current_usage + points > self.max_points:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            self.current_usage += points
            return True

    def release(self, points: int):
        with self.condition:
            self.current_usage -= points
            if self.current_usage < 0:
                self.current_usage = 0
            self.condition.notify_all()
