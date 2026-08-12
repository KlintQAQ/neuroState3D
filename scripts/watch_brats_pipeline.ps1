param(
    [string]$Root = "D:\BrainMVP-neurostate3d",
    [int]$IntervalSeconds = 300,
    [int]$MaxWorkers = 4,
    [int]$Retries = 50,
    [int]$RetrySleepSeconds = 180
)

$ErrorActionPreference = "Continue"
$DataRoot = Join-Path $Root "data"
$RawRoot = Join-Path $DataRoot "raw\BraTS2023_HF"
$ModelReadyRoot = Join-Path $DataRoot "model_ready\BraTS2023_HF_128"
$LogDir = Join-Path $DataRoot "logs\downloads\BraTS2023_HF"
$CacheDir = Join-Path $DataRoot "cache\huggingface"
$WatchLog = Join-Path $LogDir "watchdog_events.jsonl"
$WatchState = Join-Path $LogDir "watchdog_state.json"
$ActivePipeline = Join-Path $LogDir "active_pipeline.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

function Write-Event {
    param([hashtable]$Event)
    $Event["time"] = (Get-Date).ToString("s")
    $Event | ConvertTo-Json -Depth 8 -Compress | Add-Content -Path $WatchLog -Encoding UTF8
}
function Get-TreeStats {
    param([string]$Path)
    $files = Get-ChildItem -Recurse -File $Path -ErrorAction SilentlyContinue
    $measure = $files | Measure-Object -Property Length -Sum
    [pscustomobject]@{
        files = [int64]$measure.Count
        bytes = [int64]$measure.Sum
        gb = [math]::Round(([double]$measure.Sum / 1GB), 3)
        nii_gz = ($files | Where-Object { $_.Name -like "*.nii.gz" } | Measure-Object).Count
    }
}

function Get-DatasetStats {
    $rows = @()
    foreach ($name in @("GLI", "MEN", "PED")) {
        $path = Join-Path $RawRoot $name
        $stats = Get-TreeStats -Path $path
        $subjects = Get-ChildItem -Recurse -Directory $path -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "BraTS-*" } |
            Measure-Object
        $rows += [pscustomobject]@{
            dataset = $name
            files = $stats.files
            gb = $stats.gb
            nii_gz = $stats.nii_gz
            subjects = $subjects.Count
        }
    }
    $rows
}

function Get-ManagedProcesses {
    $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    [pscustomobject]@{
        download = @($processes | Where-Object {
            $_.Name -match "^python" -and $_.CommandLine -match "scripts\\download_brats_hf.py"
        })
        prepare = @($processes | Where-Object {
            $_.Name -match "^python" -and $_.CommandLine -match "scripts\\prepare_brats_model_ready.py"
        })
        pipeline = @($processes | Where-Object {
            $_.Name -match "^powershell" -and $_.CommandLine -match "download_brats_hf.py --datasets gli men ped"
        })
    }
}

function Get-LatestDoneMeta {
    $done = Get-ChildItem -File $LogDir -Filter "pipeline_done_*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $done) {
        return $null
    }
    try {
        return Get-Content $done.FullName -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Start-BratsPipeline {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $downloadStdout = Join-Path $LogDir "stdout_watchdog_$stamp.log"
    $downloadStderr = Join-Path $LogDir "stderr_watchdog_$stamp.log"
    $prepareStdout = Join-Path $LogDir "prepare_stdout_watchdog_$stamp.log"
    $prepareStderr = Join-Path $LogDir "prepare_stderr_watchdog_$stamp.log"
    $pipelineStdout = Join-Path $LogDir "pipeline_stdout_watchdog_$stamp.log"
    $pipelineStderr = Join-Path $LogDir "pipeline_stderr_watchdog_$stamp.log"
    $doneMeta = Join-Path $LogDir "pipeline_done_watchdog_$stamp.json"

    $script = @"
`$ErrorActionPreference = 'Continue'
`$root = '$Root'
`$dataRoot = '$DataRoot'
`$cacheDir = '$CacheDir'
`$downloadStdout = '$downloadStdout'
`$downloadStderr = '$downloadStderr'
`$prepareStdout = '$prepareStdout'
`$prepareStderr = '$prepareStderr'
`$doneMeta = '$doneMeta'
`$env:HF_HOME = `$cacheDir
`$env:HF_XET_CACHE = Join-Path `$cacheDir 'xet'
Set-Location `$root
& python scripts\download_brats_hf.py --datasets gli men ped --data-root `$dataRoot --max-workers $MaxWorkers --retries $Retries --retry-sleep $RetrySleepSeconds > `$downloadStdout 2> `$downloadStderr
`$downloadExit = `$LASTEXITCODE
`$prepareExit = `$null
if (`$downloadExit -eq 0) {
  & python scripts\prepare_brats_model_ready.py --raw-root (Join-Path `$dataRoot 'raw\BraTS2023_HF') --output-root (Join-Path `$dataRoot 'model_ready\BraTS2023_HF_128') --manifest-dir (Join-Path `$dataRoot 'manifests\BraTS2023_HF') --target-shape 128 128 128 --workers 4 > `$prepareStdout 2> `$prepareStderr
  `$prepareExit = `$LASTEXITCODE
}
[ordered]@{
  status = 'PIPELINE_DONE'
  download_exit_code = `$downloadExit
  prepare_exit_code = `$prepareExit
  download_stdout = `$downloadStdout
  download_stderr = `$downloadStderr
  prepare_stdout = `$prepareStdout
  prepare_stderr = `$prepareStderr
  finished_at = (Get-Date).ToString('s')
} | ConvertTo-Json -Depth 5 | Set-Content -Path `$doneMeta -Encoding UTF8
"@

    $process = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $script) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $pipelineStdout `
        -RedirectStandardError $pipelineStderr `
        -WindowStyle Hidden `
        -PassThru

    $summary = [ordered]@{
        status = "STARTED_BY_WATCHDOG"
        root = $Root
        data_root = $DataRoot
        pipeline_pid = $process.Id
        raw_root = $RawRoot
        model_ready_root = $ModelReadyRoot
        download_stdout = $downloadStdout
        download_stderr = $downloadStderr
        prepare_stdout = $prepareStdout
        prepare_stderr = $prepareStderr
        done_meta = $doneMeta
        max_workers = $MaxWorkers
        retries = $Retries
        retry_sleep_seconds = $RetrySleepSeconds
        started_at = (Get-Date).ToString("s")
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -Path $ActivePipeline -Encoding UTF8
    Write-Event @{ event = "pipeline_started"; pipeline_pid = $process.Id; done_meta = $doneMeta }
}

Write-Event @{ event = "watchdog_started"; interval_seconds = $IntervalSeconds; root = $Root }

while ($true) {
    $managed = Get-ManagedProcesses
    $doneMeta = Get-LatestDoneMeta
    $datasets = @(Get-DatasetStats)
    $rawStats = Get-TreeStats -Path $RawRoot
    $modelStats = Get-TreeStats -Path $ModelReadyRoot

    $state = [ordered]@{
        status = "WATCHING"
        time = (Get-Date).ToString("s")
        download_processes = @($managed.download).Count
        prepare_processes = @($managed.prepare).Count
        pipeline_processes = @($managed.pipeline).Count
        raw_gb = $rawStats.gb
        raw_files = $rawStats.files
        model_ready_gb = $modelStats.gb
        model_ready_files = $modelStats.files
        datasets = $datasets
        latest_done = $doneMeta
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -Path $WatchState -Encoding UTF8
    Write-Event @{
        event = "watchdog_tick"
        download_processes = @($managed.download).Count
        prepare_processes = @($managed.prepare).Count
        pipeline_processes = @($managed.pipeline).Count
        raw_gb = $rawStats.gb
        model_ready_gb = $modelStats.gb
    }

    $isComplete = $false
    if ($doneMeta -and $doneMeta.download_exit_code -eq 0 -and $doneMeta.prepare_exit_code -eq 0) {
        $isComplete = $true
    }

    if ($isComplete) {
        Write-Event @{ event = "complete_detected"; raw_gb = $rawStats.gb; model_ready_gb = $modelStats.gb }
        break
    }

    if (@($managed.download).Count -eq 0 -and @($managed.prepare).Count -eq 0 -and @($managed.pipeline).Count -eq 0) {
        Write-Event @{ event = "no_managed_process_restart"; raw_gb = $rawStats.gb; model_ready_gb = $modelStats.gb }
        Start-BratsPipeline
    }

    Start-Sleep -Seconds $IntervalSeconds
}
