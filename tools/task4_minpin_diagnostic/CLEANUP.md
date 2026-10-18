# Diagnostic controller teardown

This branch contains diagnostic tooling, not a production offload fix.
The cleanup change does not change the model runner, pin budget, or transfer path.

## Failure and correction

A controller exit marker does not prove that its subprocesses have exited.
With `exec > >(tee -a controller.log) 2>&1`, killing a watchdog shell can
leave its `sleep` child holding the log pipe open. The process substitution
shell then waits for `tee`, even after the main controller exits successfully.

`controller_cleanup.sh.example` uses a direct log file instead of process
substitution. Its watchdog is one Python process; on expiry, `exec` replaces
that process with the stop command. Cleanup terminates and waits for that
child before writing completion records. The watchdog accepts only the owned
diagnostic container name.

The example is an integration reference, not an approved deployment command.
Before using it, review the reservation, unique attempt, output mount, source
identity checks and the attempt-specific paths in the prepared scripts.
It must not be run against an old reservation. Existing capacity guards remain mandatory.

## CPU checks

From the repository root:

```bash
python3 -m unittest discover -s tools/task4_minpin_diagnostic \
  -p test_watchdog.py -v
bash -n tools/task4_minpin_diagnostic/controller_cleanup.sh.example
# Linux CPU-only old/fixed subprocess comparison:
python3 tools/task4_minpin_diagnostic/reproduce_cleanup.py
```

The tests replace Docker with a local stub: cancellation must prevent a later
action, expiry must target only the owned container, and other container names
must be rejected. No test starts a model or accesses an accelerator.

The controller and logger process tree must also be checked after cleanup;
neither an exit file nor a stopped container substitutes for that check.
Preserve original logs and distinguish controller exit status from a container
terminated by the controller. Hardware regression, production capacity,
precision, transfer and performance acceptance remain separate requirements.

Private run logs, measurements, resource snapshots and delivery reports are
intentionally excluded from this branch.
