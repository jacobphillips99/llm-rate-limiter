#!/usr/bin/env python3
"""Terminal-based monitoring tool for API rate limits."""

import curses
import json
import os
import time
import typing as t

from llm_rate_limiter.constants import RATE_LIMIT_STATS_PATH


class RateLimitUI:
    """Terminal UI for monitoring rate limits."""

    def __init__(self, stats_path: str = RATE_LIMIT_STATS_PATH, refresh_rate: float = 0.5) -> None:
        """Initialize the rate monitor.

        Args:
            stats_path: Path to the stats file
            refresh_rate: How often to refresh the display (seconds)
        """
        self.stats_path = stats_path
        self.refresh_rate = refresh_rate
        self.last_stats: t.Optional[dict] = None
        self.last_update_time: t.Optional[float] = None

    def _read_stats(self) -> dict[str, t.Any]:
        """Read stats from file."""
        try:
            if not os.path.exists(self.stats_path):
                return {"error": f"Stats file not found: {self.stats_path}"}

            with open(self.stats_path, "r") as f:
                data = json.load(f)

            if not data:
                return {"error": "Empty stats file"}

            self.last_update_time = data.get("timestamp", time.time())
            stats = data.get("stats", {})
            if not isinstance(stats, dict):
                return {"error": "Stats must be a dictionary"}
            return stats
        except Exception as e:
            return {"error": f"Failed to read stats: {e}"}

    def _format_bar(self, percent: float, width: int = 20) -> str:
        """Format a progress bar.

        Args:
            percent: Percentage (0-100)
            width: Width of the bar in characters

        Returns:
            Formatted progress bar string
        """
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def _format_percent(self, percent: float) -> str:
        """Format a percentage with color indication."""
        if percent < 70:
            return f"{percent:.1f}%"
        elif percent < 90:
            return f"{percent:.1f}%"  # Would be yellow in color terminal
        else:
            return f"{percent:.1f}%"  # Would be red in color terminal

    def _safe_addstr(self, stdscr: t.Any, y: int, x: int, text: str, *args: t.Any) -> bool:
        """Safely add a string to the screen, checking boundaries.

        Args:
            stdscr: Curses window
            y: Y position
            x: X position
            text: Text to add
            *args: Additional arguments for addstr

        Returns:
            True if the string was added successfully, False otherwise
        """
        try:
            max_y, max_x = stdscr.getmaxyx()
            if y >= max_y or x >= max_x:
                return False
            if x + len(text) > max_x:
                text = text[: max_x - x]
            stdscr.addstr(y, x, text, *args)
            return True
        except curses.error:
            return False

    def _draw_header(self, stdscr: t.Any, y: int) -> int:
        """Draw the header section.

        Args:
            stdscr: Curses window
            y: Starting y position

        Returns:
            Next y position
        """
        max_y, max_x = stdscr.getmaxyx()
        if y >= max_y:
            return y

        # Title
        self._safe_addstr(stdscr, y, 0, "API Rate Limit Monitor", curses.A_BOLD)
        y += 1

        # Last update time
        if self.last_update_time and y < max_y:
            time_str = time.strftime("%H:%M:%S", time.localtime(self.last_update_time))
            update_str = f"Last Update: {time_str}"
            self._safe_addstr(stdscr, y, 0, update_str)
        y += 2

        # Check if we have enough space for headers
        if y + 2 >= max_y:
            return y

        # Column headers
        self._safe_addstr(stdscr, y, 0, "PROVIDER/MODEL", curses.A_BOLD)
        self._safe_addstr(stdscr, y, 25, "REQUESTS/MIN", curses.A_BOLD)
        self._safe_addstr(stdscr, y, 50, "TOKENS/MIN", curses.A_BOLD)
        y += 1

        # Separator
        self._safe_addstr(stdscr, y, 0, "─" * (max_x - 1))
        y += 1

        return y

    def _draw_model_stats(
        self, stdscr: t.Any, y: int, provider: str, model: str, stats: dict
    ) -> int:
        """Draw stats for a single model.

        Args:
            stdscr: Curses window
            y: Starting y position
            provider: Provider name
            model: Model name
            stats: Stats for this model

        Returns:
            Next y position
        """
        max_y, max_x = stdscr.getmaxyx()
        if y >= max_y:
            return y

        # Provider/model
        model_name = f"{provider}/{model}"
        if len(model_name) > 24:
            model_name = model_name[:21] + "..."
        self._safe_addstr(stdscr, y, 0, model_name)

        # RPM
        rpm_stats = stats.get("requests_per_minute", {})
        rpm_current = rpm_stats.get("current", 0)
        rpm_limit = rpm_stats.get("limit", 0)
        rpm_percent = rpm_stats.get("percent", 0)

        rpm_str = f"{rpm_current}/{rpm_limit}"
        self._safe_addstr(stdscr, y, 25, rpm_str)

        if rpm_limit > 0:
            bar = self._format_bar(rpm_percent)
            percent = self._format_percent(rpm_percent)
            self._safe_addstr(stdscr, y, 25 + len(rpm_str) + 1, f"{bar} {percent}")

        # TPM
        tpm_stats = stats.get("tokens_per_minute", {})
        tpm_current = tpm_stats.get("current", 0)
        tpm_limit = tpm_stats.get("limit", 0)
        tpm_percent = tpm_stats.get("percent", 0)

        tpm_str = f"{tpm_current}/{tpm_limit}"
        self._safe_addstr(stdscr, y, 50, tpm_str)

        if tpm_limit > 0:
            bar = self._format_bar(tpm_percent)
            percent = self._format_percent(tpm_percent)
            self._safe_addstr(stdscr, y, 50 + len(tpm_str) + 1, f"{bar} {percent}")

        return y + 1

    def _draw_screen(self, stdscr: t.Any) -> None:
        """Draw the main screen.

        Args:
            stdscr: Curses window
        """
        stdscr.clear()
        y = 0

        # Read latest stats
        stats = self._read_stats()

        # Draw header
        y = self._draw_header(stdscr, y)

        # Handle errors
        if "error" in stats:
            self._safe_addstr(stdscr, y, 0, f"Error: {stats['error']}")
            stdscr.refresh()
            return

        # Draw provider/model stats
        for provider, provider_stats in stats.items():
            for model, model_stats in provider_stats.items():
                if "error" not in model_stats:
                    y = self._draw_model_stats(stdscr, y, provider, model, model_stats)
                    if y >= stdscr.getmaxyx()[0]:
                        break

        # Footer
        max_y, _ = stdscr.getmaxyx()
        if max_y > 0:
            self._safe_addstr(stdscr, max_y - 1, 0, "Press 'q' to quit, 'r' to refresh")

        stdscr.refresh()

    def run(self, stdscr: t.Any) -> None:
        """Run the monitoring interface.

        Args:
            stdscr: Curses window
        """
        # Setup
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(int(self.refresh_rate * 1000))

        # Main loop
        while True:
            self._draw_screen(stdscr)

            # Handle input
            try:
                key = stdscr.getkey()
                if key == "q":
                    break
                elif key == "r":
                    # Force refresh
                    continue
            except curses.error:
                # No input, continue
                pass
