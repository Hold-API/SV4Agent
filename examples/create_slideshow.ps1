param(
    [Parameter(Mandatory = $true)][string]$PhotoDirectory,
    [Parameter(Mandatory = $true)][string]$AudioPath,
    [string]$TimelinePath = (Join-Path $PSScriptRoot "concat_20261012.txt"),
    [string]$ServerUrl = "http://127.0.0.1:8000",
    [string]$OutputPath = ".\slideshow.mp4",
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"
$timelineFile = (Resolve-Path -LiteralPath $TimelinePath).Path
$audioFile = (Resolve-Path -LiteralPath $AudioPath).Path
$photoRoot = (Resolve-Path -LiteralPath $PhotoDirectory).Path

# This convenience script accepts single-quoted file paths as in the example.
# Complex ffconcat quoting can still be uploaded directly with curl.exe.
$photoNames = @()
foreach ($line in Get-Content -LiteralPath $timelineFile -Encoding UTF8) {
    if ($line -match '^\s*file\s+') {
        $fileEntry = [regex]::Match($line, "^\s*file\s+'([^']+)'\s*(?:#.*)?$")
        if (-not $fileEntry.Success) {
            throw "Use single-quoted file paths in the timeline, or upload with curl.exe directly."
        }
        $photoNames += [System.IO.Path]::GetFileName($fileEntry.Groups[1].Value)
    }
}
$photoNames = @($photoNames | Select-Object -Unique)
if ($photoNames.Count -eq 0) { throw "No image file entries in timeline." }

$curlArguments = @(
    "--fail-with-body", "--show-error",
    ($ServerUrl.TrimEnd('/') + "/v1/video/slideshow"),
    "-F", ('timeline=@"' + $timelineFile + '"'),
    "-F", ('audio=@"' + $audioFile + '"'),
    "-F", "codec=h264", "-F", "preset=p7", "-F", "quality=23",
    "-F", "width=1920", "-F", "height=1080", "-F", "fps=24",
    "-F", "audio_codec=copy"
)
if ($ApiKey) { $curlArguments += @("-H", "Authorization: Bearer $ApiKey") }
foreach ($name in $photoNames) {
    $photo = (Resolve-Path -LiteralPath (Join-Path $photoRoot $name)).Path
    $curlArguments += @("-F", ('images=@"' + $photo + '"'))
}

# Keep an HTTP error body separate from the requested MP4 output.
$temporaryOutput = $OutputPath + "." + [guid]::NewGuid().ToString("N") + ".part"
try {
    & curl.exe @curlArguments --output $temporaryOutput
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $temporaryOutput) {
            Get-Content -LiteralPath $temporaryOutput -Raw | Write-Host
        }
        throw "Slideshow request failed (curl exit $LASTEXITCODE)."
    }
    Move-Item -LiteralPath $temporaryOutput -Destination $OutputPath -Force
    Write-Host "Saved: $OutputPath"
} finally {
    if (Test-Path -LiteralPath $temporaryOutput) { Remove-Item -LiteralPath $temporaryOutput -Force }
}
