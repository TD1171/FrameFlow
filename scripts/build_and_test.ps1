# Single entry point: configure, build, and run every check.
# Exits non-zero if ANY check fails or is skipped, so it is safe to use as a
# gate before claiming that something works.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_and_test.ps1
#
# Optional: -Clean  wipes the build directory first.

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo "build"
$scratch = Join-Path $repo "build\_test"
$failures = New-Object System.Collections.Generic.List[string]
$script:blocked = $false

# Steps run in dependency order. Once something essential fails, everything
# downstream is SKIPPED and counted as a failure rather than executed against a
# broken or stale tree -- previously a failed build still let later steps
# "pass", so the suite reported success while the build reported failure.
function Step($name, [scriptblock]$body, [switch]$Essential) {
    Write-Host ""
    if ($script:blocked) {
        Write-Host "=== $name ===" -ForegroundColor DarkGray
        Write-Host "SKIP: $name (a prerequisite failed)" -ForegroundColor Yellow
        $script:failures.Add("$name (skipped)")
        return
    }
    Write-Host "=== $name ===" -ForegroundColor Cyan
    try {
        & $body
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        Write-Host "PASS: $name" -ForegroundColor Green
    } catch {
        Write-Host "FAIL: $name -- $($_.Exception.Message)" -ForegroundColor Red
        $script:failures.Add($name)
        if ($Essential) { $script:blocked = $true }
    }
}

# MSVC is not on PATH by default; vcvars64.bat puts cl.exe, the Windows SDK and
# the linker in scope. nvcc also needs cl.exe reachable, so every build runs
# inside this environment.
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

function InVcEnv($cmdline) {
    cmd /c "call `"$vcvars`" >nul 2>&1 && cd /d `"$repo`" && $cmdline"
}

if ($Clean -and (Test-Path $build)) {
    Write-Host "removing $build"
    Remove-Item $build -Recurse -Force
}

$exe = Join-Path $build "frameflow.exe"

# Delete the previous binary before building, so a failed build cannot leave the
# last good executable in place for downstream checks to pass against.
if (Test-Path $exe) { Remove-Item $exe -Force }

Step "configure" { InVcEnv "cmake -S . -B build -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release" } -Essential
Step "build"     { InVcEnv "cmake --build build" } -Essential

Step "binary exists" {
    if (-not (Test-Path $exe)) { throw "frameflow.exe was not produced" }
    $global:LASTEXITCODE = 0
} -Essential

# Every run gets a fresh scratch directory. A fixed output path that is only
# cleaned at the end lets a stale file from the previous run satisfy a
# Test-Path check, and makes the encoder's truncating open more failure-prone.
if (Test-Path $scratch) { Remove-Item $scratch -Recurse -Force }
New-Item -ItemType Directory -Path $scratch | Out-Null

# Correctness at two resolutions. 320x240 is an exact multiple of the 16x16
# thread block; 1000x996 deliberately is not (62.5 and 62.25 blocks), so the
# last block in each dimension runs threads past the image edge and the bounds
# checks are actually exercised. Clean dimensions hide off-by-one bugs.
#
# Odd dimensions such as 1001x997 cannot be used here: the mp4 encoder requires
# even width and height and silently rounds down, so the test would report a
# resolution it never actually ran. Even-but-unaligned gives the same coverage
# without lying about what was tested.
foreach ($res in @(@{w=320; h=240; tag="aligned 320x240"},
                   @{w=1000; h=996; tag="unaligned 1000x996"})) {

    $clip = Join-Path $scratch "clip_$($res.w)x$($res.h).mp4"
    $out  = Join-Path $scratch "out_$($res.w)x$($res.h).mp4"

    Step "generate clip ($($res.tag))" {
        python (Join-Path $repo "scripts\generate_test_video.py") `
            --output $clip --width $res.w --height $res.h --frames 6 --fps 30 | Out-Null
        if (-not (Test-Path $clip)) { throw "generator produced no file" }
    } -Essential

    Step "runs and writes output ($($res.tag))" {
        & $exe --input $clip --output $out --frames 4 --warmup 2 --save-frames | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "frameflow exited $LASTEXITCODE" }
        if (-not (Test-Path $out)) { throw "no output video was written" }
    }

    # The check that actually reads pixels. Without it, a kernel writing all
    # zeros or swapping B and R passes every other check in this file.
    Step "stage output matches NumPy oracle ($($res.tag))" {
        python (Join-Path $repo "scripts\verify_stages.py") --dir (Join-Path $repo "results")
        if ($LASTEXITCODE -ne 0) { throw "stage verification failed" }
    }
}

# Negative cases. Each asserts the binary REJECTED its input. The positive cases
# above (which assert exit 0 and correct pixels) are the control proving these
# are not passing merely because the binary is broken.
#
# These run through cmd rather than PowerShell's call operator on purpose: with
# $ErrorActionPreference = "Stop", any native process writing to stderr raises a
# terminating PowerShell error, which would make a correctly-rejecting binary
# look like a failed test. cmd reports only the process exit code.
#
# Two guards keep this from being a tautology:
#   1. The executable must exist. Otherwise every rejection test passes
#      trivially when the build failed -- cmd also returns non-zero for a
#      command it cannot find.
#   2. The exit code must be exactly 1 (EXIT_FAILURE). "Any non-zero" would
#      accept 9009 (command not found) and crash codes, so a segfaulting binary
#      would pass a test named "rejects bad input".
function ExpectRejected($argline, $what) {
    if (-not (Test-Path $exe)) { throw "cannot test rejection: $exe does not exist" }
    cmd /c "`"$exe`" $argline >nul 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 1) { throw "expected exit code 1 $what, got $code" }
    $global:LASTEXITCODE = 0
}

Step "rejects a missing input file" { ExpectRejected "--input data\definitely_not_here.mp4" "for a missing file" }
Step "rejects an unknown flag"      { ExpectRejected "--input x.mp4 --not-a-real-flag" "for an unknown flag" }
Step "rejects a missing --input"    { ExpectRejected "" "when --input is absent" }
Step "rejects an even --ksize"      { ExpectRejected "--input x.mp4 --ksize 4" "for an even kernel size" }
Step "rejects a negative --sigma"   { ExpectRejected "--input x.mp4 --sigma -1" "for a negative sigma" }

if (Test-Path $scratch) { Remove-Item $scratch -Recurse -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "========================================"
if ($failures.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "========================================"
    exit 0
} else {
    Write-Host "$($failures.Count) CHECK(S) FAILED:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "========================================"
    exit 1
}
