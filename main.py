"""
Kitsune-Agent Entry Point.
Starts the Telegram bot and initializes all subsystems.
"""

import logging
import sys

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


def main():
    """Main application entry point."""
    print("🦊 Starting Kitsune-Agent...")

    try:
        # Load configuration
        config = Config()
        setup_logging(config.log_level)

        # Initialize and run bot
        bot = KitsuneBot(config)
        bot.run()

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please check your .env file and ensure required keys are set.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🦊 Kitsune-Agent stopped gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
