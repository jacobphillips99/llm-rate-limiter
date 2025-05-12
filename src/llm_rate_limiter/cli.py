import curses
from dataclasses import dataclass

import draccus

from llm_rate_limiter.constants import MONITOR_REFRESH_RATE, RATE_LIMIT_STATS_PATH
from llm_rate_limiter.ui import RateLimitUI


@dataclass
class MonitorConfig:
    """Configuration for the rate monitor as deployed in the CLI."""

    stats_path: str = RATE_LIMIT_STATS_PATH
    refresh_rate: float = MONITOR_REFRESH_RATE


@draccus.wrap()
def main(cfg: MonitorConfig) -> None:
    """
    Monitor API rate limits

    Args:
        cfg: Monitor configuration
    """
    # Run with curses
    monitor = RateLimitUI(cfg.stats_path, cfg.refresh_rate)
    curses.wrapper(monitor.run)


if __name__ == "__main__":
    main()
