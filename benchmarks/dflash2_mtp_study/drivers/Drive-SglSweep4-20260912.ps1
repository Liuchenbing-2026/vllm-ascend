# Third leg: nospec and mtp3 re-run carrying the EOS-respected cells.
#
# Why this is needed. --ignore-eos was MY choice, not SGLang's, and 20260911ak showed it is not a
# neutral one: DFlash K7's advance goes 4.21 -> 5.31 (C1) and 3.90 -> 5.18 (C16) once EOS is
# respected, i.e. my own kou-jing was suppressing DFlash by 26-33%. K3 only gained 13-15%, so the
# bias is NOT uniform across arms and cannot be waved off as a common offset. Until nospec and
# mtp3 are measured with the same stop condition, the EOS-respected cells have a numerator and
# no denominator, and no ratio there may be quoted.
#
# Uses the same terminal-marker waiting as driver 2 (.done/.failed, not .out).

$ErrorActionPreference = 'Stop'
$queue = 'C:\Users\wzy85\tq_dsv4_scripts\queue_dsv4_repro18'
$work  = 'C:\Users\wzy85\Documents\Codex\2026-08-21\14171pr-0-26\work'
$pub   = 'C:\Users\wzy85\tq_dsv4_scripts\Publish-Dsv4ReproCommand.ps1'
$log   = Join-Path $work 'drive-sgl4-20260912.log'

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
  @{ id='20260912a'; script='benchserve_start_k7_sglthink_m18_20260912.sh';   marker='BENCHSERVE_START_READY'; wait=35 },
  @{ id='20260912b'; script='benchserve_bench_k7_sglthink_m18_20260912.sh';   marker='BENCHSGL_PASS';          wait=30 },
  @{ id='20260912c'; script='benchserve_start_mtp3_sglthink_m18_20260912.sh'; marker='BENCHSERVE_START_READY'; wait=35 },
  @{ id='20260912d'; script='benchserve_bench_mtp3_sglthink_m18_20260912.sh'; marker='BENCHSGL_PASS';          wait=30 }
)

Say 'waiting on 20260911ao (nospec-eos) to release the cards'
$prev = Wait-Task '20260911ao' 25
if ($null -eq $prev) { Say 'ABORT: 20260911ak never reached a terminal state'; exit 1 }
Say ('ao settled: ' + (($prev.text -split "`n" | Where-Object { $_ -match 'BENCHSGL|released' } | Select-Object -Last 3) -join ' | '))

foreach ($s in $steps) {
  Publish $s.script $s.id
  $res = Wait-Task $s.id $s.wait
  if ($null -eq $res) { Say ("ABORT: {0} timed out after {1} min" -f $s.id, $s.wait); exit 1 }
  if ($res.text -match $s.marker) { Say ("OK {0} ({1})" -f $s.id, $s.marker); continue }
  Say ("FAILED {0}: marker {1} absent" -f $s.id, $s.marker)
  Say (($res.text -split "`n" | Select-Object -Last 6) -join ' | ')
  exit 1
}

Say 'CHAIN 4 COMPLETE'
