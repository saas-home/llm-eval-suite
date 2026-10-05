import sys
import os

# Add parent directory to sys.path if not present so env_loader is discoverable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import env_loader
except ImportError:
    pass
