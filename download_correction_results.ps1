param(
    [Parameter(Mandatory=$true)][string]$ArchiveUri,
    [string]$DestinationDirectory = "$env:USERPROFILE\Downloads",
    [int]$PollSeconds = 60,
    [int]$TimeoutHours = 72
)
# Run on your laptop, not inside Workbench. Requires an authenticated gcloud CLI.
# The VM uploads to a private bucket before stopping; this script waits for that
# exact object and downloads it once. It never changes IAM or deletes files.
$ErrorActionPreference = "Stop"
if ($ArchiveUri -notmatch '^gs://[^/]+/.+_analysis\.tar\.gz$') {
    throw "Provide the exact gs://.../SUITE_analysis.tar.gz printed/derived from the run."
}
if ($PollSeconds -lt 5 -or $TimeoutHours -lt 1) { throw "Invalid polling interval or timeout." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "Install and authenticate gcloud first." }
if (-not (Test-Path -LiteralPath $DestinationDirectory -PathType Container)) {
    throw "Destination directory does not exist."
}
$resultName = ($ArchiveUri -split '/')[-1]
$resultPath = Join-Path (Resolve-Path -LiteralPath $DestinationDirectory).Path $resultName
if (Test-Path -LiteralPath $resultPath) { throw "Refusing to overwrite $resultPath" }
$deadline = (Get-Date).AddHours($TimeoutHours)
while ((Get-Date) -lt $deadline) {
    # gcloud writes diagnostics for missing objects. Keep them visible so auth
    # or bucket permission problems are not misrepresented as a completed run.
    & gcloud storage objects describe $ArchiveUri --format="value(size)"
    if ($LASTEXITCODE -eq 0) {
        & gcloud storage cp --no-clobber $ArchiveUri $resultPath
        if ($LASTEXITCODE -ne 0) { throw "Download failed; retry the copy command." }
        Write-Host "Downloaded analysis archive: $resultPath"
        exit 0
    }
    Write-Host "Object not available yet (or access failed); retrying in $PollSeconds seconds."
    Start-Sleep -Seconds $PollSeconds
}
throw "Timed out. The remote archive, if created later, remains in Cloud Storage."
