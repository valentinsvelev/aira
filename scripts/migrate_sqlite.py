"""One-time upgrade CLI; target is the container's DATABASE_URL."""
import argparse
import os
from pathlib import Path

from aira.migrate import migrate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite', type=Path, required=True)
    args = parser.parse_args()
    print(migrate(args.sqlite, os.environ['DATABASE_URL']))
