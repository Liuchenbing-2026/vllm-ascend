# Second leg of the SGLang-kou-jing sweep: k3 (matched verify width vs MTP3), then k7 with EOS
# respected (the one kou-jing axis I chose myself rather than inherited).
#
# Fix vs driver 1: wait on the queue's TERMINAL MARKERS (.done / .failed), not only on .out.
# Driver 1 declared a 30-minute timeout ~48 minutes after publishing 20260911af, which means the
# .out-only poll was not tracking the task's actual state. .done/.failed are written by the queue
# runner itself, so they are the authoritative signal.

$ErrorActionPreference = 'Stop'
$queue = 'C:\Users\wzy85\tq_dsv4_scripts\queue_dsv4_repro18'
$work  = 'C:\Users\wzy85\Documents\Codex\2026-08-21\14171pr-0-26\work'
$pub   = 'C:\Users\wzy85\tq_dsv4_scripts\Publish-Dsv4ReproCommand.ps1'
$log   = Join-Path $work 'drive-sgl2-20260911.log'

function Say($msg) {
  $line = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

function Wait-Task($id, $maxMinutes) {
  $deadline = (Get-Date).AddMinutes($maxMinutes)
  while ((Get-Date) -lt $deadline) {
    $done   = Test-Path (Join-Path $queue "$id.done")
    $failed = Test-Path (Join-Path $queue "$id.failed")
    if ($done -or $failed) {
      Start-Sleep -Seconds 5
      $o = Join-Path $queue "$id.out"
      $text = ''
      if (Test-Path $o) { $text = Get-Content $o -Raw -Encoding utf8 }
      return @{ done = $done; failed = $failed; text = $text }
    }
    Start-Sleep -Seconds 15
  }
  return $null
}

function Publish($script, $id) {
  Say "publish $id <- $script"
  & $pub -SourcePath (Join-Path $work $script) -HostId 18 -Id $id | Out-Null
}

$steps = @(
  @{ id='20260911ah'; script='benchserve_start_k3_sgl_m18_20260911b.sh';      marker='BENCHSERVE_START_READY'; wait=35 },
  @{ id='20260911ai'; script='benchserve_bench_k3_sgl_m18_20260911b.sh';      marker='BENCHSGL_PASS';          wait=30 },
  @{ id='20260911aj'; script='benchserve_start_k7eos_sgl_m18_20260911c.sh';   marker='BENCHSERVE_START_READY'; wait=35 },
  @{ id='20260911ak'; script='benchserve_bench_k7eos_sgl_m18_20260911c.sh';   marker='BENCHSGL_PASS';          wait=30 }
)

Say 'waiting on 20260911ag (mtp7 attach/teardown) before touching the cards'
$prev = Wait-Task '20260911ag' 30
if ($null -eq $prev) { Say 'ABORT: 20260911ag never reached a terminal state'; exit 1 }
Say ('ag settled: ' + (($prev.text -split "`n" | Where-Object { $_ -match 'BENCHSGL|NOT_READY|ERROR|released' } | Select-Object -Last 4) -join ' | '))

# A start whose bench never runs leaves the cards held. Whatever ag did, reclaim before the next
# start so the k3 preflight is not waiting on a stale mtp7 container.
Publish 'release_own_cards_m18_20260910at.sh' '20260911agr'
$r = Wait-Task '20260911agr' 12
if ($null -ne $r) { Say ('reclaim: ' + (($r.text -split "`n" | Select-Object -Last 3) -join ' | ')) }

foreach ($s in $steps) {
  Publish $s.script $s.id
  $res = Wait-Task $s.id $s.wait
  if ($null -eq $res) { Say ("ABORT: {0} timed out after {1} min" -f $s.id, $s.wait); exit 1 }
  if ($res.text -match $s.marker) { Say ("OK {0} ({1})" -f $s.id, $s.marker); continue }
  Say ("FAILED {0}: marker {1} absent" -f $s.id, $s.marker)
  Say (($res.text -split "`n" | Select-Object -Last 6) -join ' | ')
  Publish 'release_own_cards_m18_20260910at.sh' ($s.id + 'r')
  Wait-Task ($s.id + 'r') 12 | Out-Null
  exit 1
}

Say 'CHAIN 2 COMPLETE'
