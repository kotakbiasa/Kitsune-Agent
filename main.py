"""
Kitsune-Agent Entry Point.
Starts the Telegram bot and initializes all subsystems.
Includes auto-restart on crash.
"""

import logging
import os
import sys
import time

from kitsune.config import Config
from kitsune.bot import KitsuneBot


def setup_logging(log_level: str):
    """Configure structured logging for the application."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress verbose logs from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def _run_once():
    """Run the bot once. Returns True if clean exit, False if should restart."""
    print("🦊 Starting Kitsune-Agent...")

    try:
        config = Config()
        setup_logging(config.log_level)

        bot = KitsuneBot(config)
        bot.run()
        return True

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please check your .env file and ensure required keys are set.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🦊 Kitsune-Agent stopped gracefully.")
        return True
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main application entry point with auto-restart loop."""
    max_restarts_env = os.getenv("AUTO_RESTART_MAX_RETRIES", "3").strip()
    restart_delay_env = os.getenv("AUTO_RESTART_DELAY_SECONDS", "5").strip()

    try:
        max_restarts = int(max_restarts_env)
    except ValueError:
        max_restarts = 3
    try:
        restart_delay = int(restart_delay_env)
    except ValueError:
        restart_delay = 5

    restart_count = 0
    while True:
        clean = _run_once()
        if clean:
            sys.exit(0)

        restart_count += 1
        if restart_count > max_restarts:
            print(f"\n💀 Max restarts ({max_restarts}) exceeded. Shutting down.")
            sys.exit(1)

        print(f"\n🔄 Restarting in {restart_delay}s... (attempt {restart_count}/{max_restarts})")
        time.sleep(restart_delay)


if __name__ == "__main__":
    main()
