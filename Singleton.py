import os
import pickle
import threading

CACHE_FILE = "precomputed_cache.pkl"


class Cache:
    _instance = None  # Single instance placeholder
    _lock = threading.Lock()  # Lock to ensure thread-safety

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:  # Ensure only one thread can access this block
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance._load_cache(*args, **kwargs)
        return cls._instance

    def _load_cache(self, *args, **kwargs):
        """Load the cache from disk if it exists."""
        print("loading cache")
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = {}

    def get_data(self):
        """Provide access to the loaded dictionary."""
        return self.data

    def save_cache(self):
        pass