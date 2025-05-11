import curses
from dataclasses import dataclass

import draccus

from llm_rate_limiter.constants import RATE_LIMIT_STATS_PATH, MONITOR_REFRESH_RATE
from llm_rate_limiter.monitor import RateMonitor

@dataclass
class MonitorConfig:
    """Configuration for the rate monitor."""

    stats_path: str = RATE_LIMIT_STATS_PATH  # Path to stats file
    refresh_rate: float = MONITOR_REFRESH_RATE  # Refresh rate in seconds


@draccus.wrap()
def main(cfg: MonitorConfig) -> None:
    """
    Monitor API rate limits

    Args:
        cfg: Monitor configuration
    """
    # Run with curses
    monitor = RateMonitor(cfg.stats_path, cfg.refresh_rate)
    curses.wrapper(monitor.run)


if __name__ == "__main__":
    main()
