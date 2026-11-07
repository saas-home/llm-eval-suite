import threading
class BoundedBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    def put(self, item):
        with self.lock:
            while len(self.items) >= self.capacity:
                self.not_full.wait()
            self.items.append(item)
            self.not_empty.notify()
    def get(self):
        with self.lock:
            while not self.items:
                self.not_empty.wait()
            item = self.items.pop(0)
            self.not_full.notify()
            return item
