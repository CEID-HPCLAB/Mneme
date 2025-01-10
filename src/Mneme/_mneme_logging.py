import logging
from enum import Enum

DEFAULT_LOGGING_LEVEL = logging.INFO
BENCH_LEVEL = 5

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    BENCHMARK = BENCH_LEVEL
    
# Custom level for benchmarking
class CustomLogger(logging.Logger):
    def benchmark(self, message, *args, **kwargs):
        if self.isEnabledFor(BENCH_LEVEL):
            self._log(BENCH_LEVEL, message, args, **kwargs)

logging.addLevelName(BENCH_LEVEL, "BENCHMARK")
logging.setLoggerClass(CustomLogger)

_Mneme_logger = logging.getLogger("__Mneme__")
_Mneme_logger.setLevel(DEFAULT_LOGGING_LEVEL)
# Custom log message format
formatter = logging.Formatter("|%(funcName)s():%(lineno)d| - %(message)s")

# Stream handler for logging to console
handler = logging.StreamHandler()
handler.setFormatter(formatter)
_Mneme_logger.addHandler(handler)