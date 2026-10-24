from __future__ import annotations

import argparse

from .live_server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plant-Health Inspection Rover — live-only dashboard (no demo or synthetic data)"
    )
    parser.add_argument("command", choices=("serve",), nargs="?", default="serve",
                        help="start the live-only dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local-only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
