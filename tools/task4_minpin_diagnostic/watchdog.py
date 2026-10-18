"""Single-process watchdog; exec replaces the process, avoiding orphan sleeps."""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()
    if args.container != "task4-cpuoffload-031-20261017":
        parser.error("Only the owned task container is allowed")
    time.sleep(max(0, args.deadline - time.time()))
    os.execvp("docker", ["docker", "stop", "--time", "10", args.container])


if __name__ == "__main__":
    main()
