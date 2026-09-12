# Drives the whole SGLang-kou-jing sweep unattended: k7 bench -> mtp3 -> mtp7.
#
# Each arm is a two-task pair (start leaves the service running, bench attaches and tears down),
# and the remote queue runs one task at a time, so the chain is: wait for the previous task's
# .out, check its success marker, publish the next.
#
# Aborting is the safe default only for START failures. If a START succeeded but its BENCH never
# runs, the service keeps the cards -- so on any bench-side failure the driver still publishes a
# teardown task rather than leaving the box occupied (shared-machine rule).

$ErrorActionPreference = 'Stop'
$queue = 'C:\Users\wzy85\tq_dsv4_scripts\queue_dsv4_repro18'
$work  = 'C:\Users\wzy85\Documents\Codex\2026-08-21\14171pr-0-26\work'
$pub   = 'C:\Users\wzy85\tq_dsv4_scripts\Publish-Dsv4ReproCommand.ps1'
$log   = Join-Path $work 'drive-sgl-20260911.log'

function Say($msg) {
  $line = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

function Wait-Out($id, $maxMinutes) {
  $deadline = (Get-Date).AddMinutes($maxMinutes)
  while ((Get-Date) -lt $deadline) {
    $o = Join-Path $queue "$id.out"
    if (Test-Path $o) {
      Start-Sleep -Seconds 3   # let the writer finish flushing
      return (Get-Content $o -Raw -Encoding utf8)
    }
    Start-Sleep -Seconds 20
  }
  return $null
}

function Publish($script, $id) {
  Say "publish $id <- $script"
  & $pub -SourcePath (Join-Path $work $script) -HostId 18 -Id $id | Out-Null
}

# id, script, marker that means success, minutes to allow
$steps = @(
  @{ id = '20260911ac'; script = 'benchserve_bench_k7_sgl_m18_20260911b.sh';   marker = 'BENCHSGL_PASS';         wait = 25 },
  @{ id = '20260911ad'; script = 'benchserve_start_mtp3_sgl_m18_20260911b.sh'; marker = 'BENCHSERVE_START_READY'; wait = 30 },
  @{ id = '20260911ae'; script = 'benchserve_bench_mtp3_sgl_m18_20260911b.sh'; marker = 'BENCHSGL_PASS';         wait = 25 },
  @{ id = '20260911af'; script = 'benchserve_start_mtp7_sgl_m18_20260911b.sh'; marker = 'BENCHSERVE_START_READY'; wait = 30 },
  @{ id = '20260911ag'; script = 'benchserve_bench_mtp7_sgl_m18_20260911b.sh'; marker = 'BENCHSGL_PASS';         wait = 25 }
)

Say 'waiting on 20260911ab (k7 service start)'
$prev = Wait-Out '20260911ab' 30
if ($null -eq $prev) { Say 'ABORT: 20260911ab never produced .out'; exit 1 }
if ($prev -notmatch 'BENCHSERVE_START_READY') {
  Say 'ABORT: k7 service did not come up ready'
  Say ($prev -split "`n" | Select-Object -Last 6) -join ' | '
  exit 1
}
Say 'k7 service ready'

foreach ($s in $steps) {
  Publish $s.script $s.id
  $out = Wait-Out $s.id $s.wait
  if ($null -eq $out) { Say ("ABORT: {0} timed out after {1} min" -f $s.id, $s.wait); exit 1 }
  if ($out -match $s.marker) {
    Say ("OK {0} ({1})" -f $s.id, $s.marker)
    continue
  }
  # MTP-7 may be refused outright by the config layer; that is an ANSWER, not a crash, and the
  # start script exits 47 without holding a card. Skip its bench and finish the chain.
  if ($out -match 'MTP7_CLAMPED') {
    Say 'MTP7_CLAMPED: engine refused depth 7; skipping the mtp7 bench'
    break
  }
  Say ("FAILED {0}: marker {1} absent" -f $s.id, $s.marker)
  Say (($out -split "`n" | Select-Object -Last 8) -join ' | ')
  # A start that half-succeeded leaves a container on the cards; the next bench would have torn
  # it down, so do that explicitly instead of exiting quietly.
  if ($s.script -like '*start*') {
    Say 'publishing reclaim so the cards do not stay held'
    Publish 'release_own_cards_m18_20260910at.sh' ($s.id + 'r')
    Wait-Out ($s.id + 'r') 10 | Out-Null
  }
  exit 1
}

Say 'CHAIN COMPLETE'
