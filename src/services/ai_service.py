import hashlib
from dataclasses import dataclass

# from dataclasses import TTLCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import ai
from src.config import settings