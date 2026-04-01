import threading

class WeightedSemaphore:
    def __init__(self, max_points: int):
        self.max_points = max_points
        self.current_usage = 0
        self.condition = threading.Condition()

    def acquire(self, points: int):
        with self.condition:
            while self.current_usage + points > self.max_points:
                self.condition.wait()
            self.current_usage += points

    def release(self, points: int):
        with self.condition:
            self.current_usage -= points
            if self.current_usage < 0:
                self.current_usage = 0
            self.condition.notify_all()
