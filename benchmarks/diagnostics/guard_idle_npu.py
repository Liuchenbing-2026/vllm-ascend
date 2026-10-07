# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail closed unless npu-smi reports no processes on every requested card.

Run on the host before a diagnostic. This is a snapshot, not a reservation;
coordinate with other users before a long benchmark. No process is stopped.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone


def check_snapshot(text, cards):
    marker = "Process id"
    if marker not in text:
        raise ValueError("npu-smi process table was not found")
    process_table = text.split(marker, 1)[1]
    idle = set()
    for line in process_table.splitlines():
        if "No running processes found in NPU" in line:
            card_text = line.split("No running processes found in NPU", 1)[1].strip(" |")
            if card_text.isdigit():
                idle.add(int(card_text))
    occupied = {}
    for line in process_table.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 6:
            continue
        device = fields[1].split()
        if len(device) == 2 and all(part.isdigit() for part in device) and fields[2].isdigit():
            card = int(device[0])
            occupied.setdefault(card, []).append({"pid": int(fields[2]), "name": fields[3]})
    unknown = sorted(set(cards) - idle - set(occupied))
    return {
        "cards": cards,
        "occupied": {str(c): occupied[c] for c in cards if c in occupied},
        "unknown": unknown,
        "safe_snapshot": not unknown and all(c in idle and c not in occupied for c in cards),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", nargs="+", type=int, required=True)
    parser.add_argument("--container", help="Use npu-smi in this existing container; IDs must match physical cards")
    args = parser.parse_args()
    command = ["npu-smi", "info"]
    if args.container:
        command = ["docker", "exec", args.container, *command]
    output = subprocess.check_output(command, text=True, timeout=30)
    report = check_snapshot(output, args.cards)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["safe_snapshot"] else 3)


if __name__ == "__main__":
    main()
