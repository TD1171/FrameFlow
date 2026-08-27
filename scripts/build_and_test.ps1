# Single entry point: configure, build, and run every check.
# Exits non-zero if ANY step fails, so it is safe to use as a gate before
# claiming that something works.
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
$failures = New-Object System.Collections.Generic.List[string]

function Step($name, [scriptblock]$body) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    try {
        & $body
        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            throw "exit code $LASTEXITCODE"
        }
        Write-Host "PASS: $name" -ForegroundColor Green
    } catch {
        Write-Host "FAIL: $name -- $($_.Exception.Message)" -ForegroundColor Red
        $script:failures.Add($name)
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

# Delete the previous binary before building. Without this, a failed build
# leaves the last successful frameflow.exe in place and every downstream check
# happily passes against stale code -- the build reports failure while the
# suite reports success, which is worse than either alone.
if (Test-Path $exe) { Remove-Item $exe -Force }

Step "configure" { InVcEnv "cmake -S . -B build -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release" }
Step "build"     { InVcEnv "cmake --build build" }

Step "binary exists" {
    if (-not (Test-Path $exe)) { throw "frameflow.exe was not produced" }
    $global:LASTEXITCODE = 0
}

# A tiny clip is enough for the smoke checks and keeps this script fast.
$clip = Join-Path $repo "data\_bt_smoke.mp4"
Step "generate smoke clip" {
    python (Join-Path $repo "scripts\generate_test_video.py") `
        --output $clip --width 320 --height 240 --frames 10 --fps 30 | Out-Null
}

Step "runs and writes output" {
    $out = Join-Path $repo "results\_bt_smoke_out.mp4"
    & $exe --input $clip --output $out --frames 5 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "frameflow exited $LASTEXITCODE" }
    if (-not (Test-Path $out)) { throw "no output video was written" }
}

# Negative cases. Each asserts a NON-zero exit, so the positive case above
# ("runs and writes output", which asserts zero) is the control that proves
# these are not passing simply because the binary is broken.
#
# These run through cmd rather than PowerShell's call operator on purpose:
# with $ErrorActionPreference = "Stop", any native process writing to stderr
# raises a terminating PowerShell error, which would make a correctly-rejecting
# binary look like a failed test. cmd reports only the process exit code, which
# is the thing actually being asserted.
# Asserts the binary rejected its input. Two guards make this a real test
# rather than a tautology:
#
#   1. The executable must exist. Without this, every rejection test passes
#      trivially when the build failed -- cmd returns non-zero for a command it
#      cannot find, which looks identical to a clean rejection.
#   2. The exit code must be exactly 1 (EXIT_FAILURE). "Any non-zero" would
#      also accept 9009 (command not found) and 0xC0000135-style crash codes,
#      so a segfaulting binary would pass a test named "rejects bad input".
function ExpectRejected($argline, $what) {
    if (-not (Test-Path $exe)) { throw "cannot test rejection: $exe does not exist" }
    cmd /c "`"$exe`" $argline >nul 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 1) { throw "expected exit code 1 $what, got $code" }
    $global:LASTEXITCODE = 0
}

Step "rejects a missing input file" {
    ExpectRejected "--input data\definitely_not_here.mp4" "for a missing file"
}

Step "rejects an unknown flag" {
    ExpectRejected "--input `"$clip`" --not-a-real-flag" "for an unknown flag"
}

Step "rejects a missing --input" {
    ExpectRejected "" "when --input is absent"
}

Remove-Item $clip, (Join-Path $repo "results\_bt_smoke_out.mp4") -ErrorAction SilentlyContinue

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
