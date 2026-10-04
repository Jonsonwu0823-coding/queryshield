[CmdletBinding()]
param(
    [string]$BailianBaseUrl,
    [string]$EvidenceRoot,
    [ValidateRange(1, 65535)][int]$PostgresPort = 5433,
    [switch]$DatabasePreflightOnly,
    [switch]$StateReplayOnly,
    [switch]$ProvenanceReplayOnly,
    [switch]$PreflightOnly,
    [switch]$R05PreflightOnly,
    [switch]$SmokeOnly,
    [switch]$T05Only,
    [string]$CandidateManifestSha256,
    [string]$FullRealSummary,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$previousLocation = Get-Location
$projectRoot = Split-Path -Parent $PSScriptRoot
$checkScript = Join-Path $PSScriptRoot "check.ps1"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$databaseOnly = $DatabasePreflightOnly -or $StateReplayOnly
$requiredNames = if ($databaseOnly) {
    @("QUERYSHIELD_DATABASE_URL")
}
else {
    @(
        "QUERYSHIELD_DATABASE_URL",
        "QUERYSHIELD_MODEL_BASE_URL",
        "QUERYSHIELD_MODEL_API_KEY",
        "QUERYSHIELD_MODEL_NAME",
        "QUERYSHIELD_EMBEDDING_BASE_URL",
        "QUERYSHIELD_EMBEDDING_API_KEY",
        "QUERYSHIELD_EMBEDDING_MODEL_NAME",
        "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
        "QUERYSHIELD_EMBEDDING_DIMENSIONS",
        "QUERYSHIELD_RERANK_URL",
        "QUERYSHIELD_RERANK_API_KEY",
        "QUERYSHIELD_RERANK_MODEL_NAME"
    )
}
$previousValues = @{}
foreach ($name in $requiredNames) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$exitCode = 0

function Read-W05Secret {
    param([string]$Prompt)
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    $pointer = [IntPtr]::Zero
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secure)
        return [Runtime.InteropServices.Marshal]::PtrToStringUni($pointer)
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($pointer)
        }
        $secure.Dispose()
    }
}

try {
    if ($T05Only -and ($DatabasePreflightOnly -or $StateReplayOnly -or $ProvenanceReplayOnly -or $PreflightOnly -or $R05PreflightOnly -or $SmokeOnly)) {
        Write-Output 'T05Only cannot be combined with another local W05 run mode.'
        $exitCode = 2
        throw [ArgumentException]::new('Conflicting T05 run modes.')
    }

    if ($T05Only -and (
        $CandidateManifestSha256 -notmatch '^[0-9a-f]{64}$' -or
        -not $FullRealSummary -or
        -not [System.IO.Path]::IsPathRooted($FullRealSummary) -or
        -not (Test-Path -LiteralPath $FullRealSummary -PathType Leaf)
    )) {
        Write-Output 'T05Only requires the frozen source manifest SHA256 and an absolute, existing full Real summary path.'
        $exitCode = 2
        throw [ArgumentException]::new('T05 candidate evidence is incomplete.')
    }

    if (($databaseOnly -and ($R05PreflightOnly -or $PreflightOnly)) -or
        ($DatabasePreflightOnly -and $StateReplayOnly) -or
        ($ProvenanceReplayOnly -and ($databaseOnly -or $PreflightOnly -or $R05PreflightOnly)) -or
        ($SmokeOnly -and ($databaseOnly -or $ProvenanceReplayOnly -or $PreflightOnly -or $R05PreflightOnly))) {
        Write-Output 'W05 run mode switches cannot be combined.'
        $exitCode = 2
        throw [ArgumentException]::new('Conflicting W05 run modes.')
    }

    if ($databaseOnly -and $BailianBaseUrl) {
        Write-Output 'Database-only preflight does not accept or use a provider URL.'
        $exitCode = 2
        throw [ArgumentException]::new('Provider URL is not applicable to a database-only preflight.')
    }

    if ($BailianBaseUrl) {
        $serviceUri = $null
        $validUri = [Uri]::TryCreate($BailianBaseUrl.Trim(), [UriKind]::Absolute, [ref]$serviceUri)
        if (-not $validUri -or $serviceUri.Scheme -ne 'https' -or
            $serviceUri.DnsSafeHost -notmatch '^ws-[a-z0-9]+\.cn-beijing\.maas\.aliyuncs\.com$' -or
            $serviceUri.AbsolutePath.TrimEnd('/') -ne '/compatible-mode/v1' -or
            $serviceUri.UserInfo -or $serviceUri.Query -or $serviceUri.Fragment -or -not $serviceUri.IsDefaultPort) {
            Write-Output 'Use the plain Beijing workspace OpenAI compatible URL copied from Bailian API Key settings; no Markdown links or credentials.'
            $exitCode = 2
            throw [ArgumentException]::new('Invalid Bailian workspace endpoint.')
        }

        # Explicit profile selection fixes only public service configuration.
        # Revision is the same provider alias used by W03, not an immutable vendor revision.
        $bailianOrigin = $serviceUri.GetLeftPart([UriPartial]::Authority)
        $profileValues = @{
            QUERYSHIELD_MODEL_BASE_URL = "$bailianOrigin/compatible-mode/v1"
            QUERYSHIELD_MODEL_NAME = 'qwen-plus'
            QUERYSHIELD_EMBEDDING_BASE_URL = "$bailianOrigin/compatible-mode/v1"
            QUERYSHIELD_EMBEDDING_MODEL_NAME = 'text-embedding-v4'
            QUERYSHIELD_EMBEDDING_MODEL_REVISION = 'text-embedding-v4'
            QUERYSHIELD_EMBEDDING_DIMENSIONS = '1024'
            QUERYSHIELD_RERANK_URL = "$bailianOrigin/compatible-api/v1/reranks"
            QUERYSHIELD_RERANK_MODEL_NAME = 'qwen3-rerank'
        }
        foreach ($name in $profileValues.Keys) {
            [Environment]::SetEnvironmentVariable($name, $profileValues[$name], 'Process')
        }

        $sharedKey = [Environment]::GetEnvironmentVariable('QUERYSHIELD_MODEL_API_KEY', 'Process')
        if ([string]::IsNullOrWhiteSpace($sharedKey) -and -not $NonInteractive) {
            $sharedKey = Read-W05Secret 'Paste the complete Bailian API Key (hidden, not a URL)'
        }
        if (-not [string]::IsNullOrWhiteSpace($sharedKey)) {
            $sharedKey = $sharedKey.Trim()
            if ($sharedKey -notmatch '^sk-[^\s*]+$') {
                Write-Output 'API Key format is invalid; copy the complete Key, not a URL, name or masked asterisks.'
                $exitCode = 2
                throw [ArgumentException]::new('Invalid API Key format.')
            }
            foreach ($name in @('QUERYSHIELD_MODEL_API_KEY', 'QUERYSHIELD_EMBEDDING_API_KEY', 'QUERYSHIELD_RERANK_API_KEY')) {
                [Environment]::SetEnvironmentVariable($name, $sharedKey, 'Process')
            }
        }
        $sharedKey = $null

        if ([string]::IsNullOrWhiteSpace($env:QUERYSHIELD_DATABASE_URL) -and -not $NonInteractive) {
            $databasePassword = Read-W05Secret 'Enter the existing local queryshield_ro password (hidden)'
            if (-not [string]::IsNullOrEmpty($databasePassword)) {
                $escapedPassword = [Uri]::EscapeDataString($databasePassword)
                $env:QUERYSHIELD_DATABASE_URL = "postgresql://queryshield_ro:${escapedPassword}@127.0.0.1:${PostgresPort}/queryshield_test"
            }
            $databasePassword = $null
            $escapedPassword = $null
        }
        Write-Output 'Bailian profile selected: qwen-plus, text-embedding-v4 (1024), qwen3-rerank. Credentials are process-only; real calls are still unverified.'
    }

    if ($databaseOnly -and
        [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("QUERYSHIELD_DATABASE_URL", "Process")) -and
        -not $NonInteractive) {
        $databasePassword = Read-W05Secret 'Enter the existing local queryshield_ro password (hidden)'
        if (-not [string]::IsNullOrEmpty($databasePassword)) {
            $escapedPassword = [Uri]::EscapeDataString($databasePassword)
            $env:QUERYSHIELD_DATABASE_URL = "postgresql://queryshield_ro:${escapedPassword}@127.0.0.1:${PostgresPort}/queryshield_test"
        }
        $databasePassword = $null
        $escapedPassword = $null
    }

    $emptyNames = @(
        foreach ($name in $requiredNames) {
            if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
                $name
            }
        }
    )
    foreach ($name in $(if (-not $NonInteractive -and -not $BailianBaseUrl -and -not $databaseOnly) { $emptyNames })) {
        $secureValue = Read-Host -Prompt "$name (hidden input)" -AsSecureString
        $pointer = [IntPtr]::Zero
        try {
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secureValue)
            $plainValue = [Runtime.InteropServices.Marshal]::PtrToStringUni($pointer)
            if (-not [string]::IsNullOrWhiteSpace($plainValue)) {
                [Environment]::SetEnvironmentVariable($name, $plainValue, "Process")
            }
        }
        finally {
            if ($pointer -ne [IntPtr]::Zero) {
                [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($pointer)
            }
            if ($null -ne $secureValue) {
                $secureValue.Dispose()
            }
            $plainValue = $null
        }
    }

    $stillMissing = @(
        foreach ($name in $requiredNames) {
            if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
                $name
            }
        }
    )
    if ($stillMissing.Count -gt 0) {
        Write-Output ("Configuration is incomplete; missing names: " + ($stillMissing -join ", "))
        $exitCode = 2
    }
    else {
        $dimensions = 0
        if (-not $databaseOnly -and
            (-not [int]::TryParse([Environment]::GetEnvironmentVariable("QUERYSHIELD_EMBEDDING_DIMENSIONS", "Process"), [ref]$dimensions) -or $dimensions -le 0)) {
            Write-Output "QUERYSHIELD_EMBEDDING_DIMENSIONS must be a positive integer."
            $exitCode = 2
        }
        else {
            $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss-fff", [Globalization.CultureInfo]::InvariantCulture)
            # -EvidenceRoot is the base directory; the run directory below reuses the name (PowerShell variables ignore case) once it has been read.
            $evidenceBase = if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) { Join-Path $projectRoot "evidence" } else { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EvidenceRoot) }
            $evidenceRoot = Join-Path $evidenceBase "W05-real-env-$stamp"
            New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null

            $shell = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
            if (-not $shell) {
                $shell = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
            }
            if (-not $shell) {
                Write-Output "No PowerShell executable is available for check.ps1."
                $exitCode = 2
            }
            else {
                Set-Location -LiteralPath $projectRoot
                if ($T05Only) {
                    $t05Evidence = Join-Path $evidenceRoot 't05'
                    $t05Script = Join-Path $PSScriptRoot 'run_w05_t05.py'
                    Write-Output 'Running the frozen W05 T05 entry; PostgreSQL and full Real evidence are verified before the hidden unseal-key prompt.'
                    & $pythonPath $t05Script `
                        --evidence-dir $t05Evidence `
                        --candidate-manifest-sha256 $CandidateManifestSha256 `
                        --full-real-summary $FullRealSummary
                    $t05ExitCode = $LASTEXITCODE
                    Write-Output "T05 evidence root: $evidenceRoot"
                    $exitCode = if ($t05ExitCode -eq 0) { 0 } elseif ($t05ExitCode -eq 2) { 2 } else { 1 }
                }
                else {
                $dbEvidence = Join-Path $evidenceRoot "preflight-db"
                $fakeEvidence = Join-Path $evidenceRoot "full-fake"
                $realEvidence = Join-Path $evidenceRoot "full-real"
                $dbArgs = @(
                    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $checkScript,
                    "-Suite", "W05", "-Mode", "fake", "-Database", "postgres",
                    "-CheckIds", "W05-DB01", "-EvidenceDir", $dbEvidence, "-PythonPath", $pythonPath
                )
                Write-Output "Running PostgreSQL preflight; no schema reset is requested."
                & $shell @dbArgs
                $dbExitCode = $LASTEXITCODE
                if ($dbExitCode -ne 0) {
                    Write-Output "Database preflight did not pass; no model, embedding, or rerank calls were started."
                    Write-Output "Evidence root: $evidenceRoot"
                    $exitCode = if ($dbExitCode -eq 2) { 2 } else { 1 }
                }
                elseif ($DatabasePreflightOnly) {
                    Write-Output "Database preflight passed; provider checks and full W05 suites were not run. Evidence root: $evidenceRoot"
                }
                elseif ($StateReplayOnly) {
                    $replayEvidence = Join-Path $evidenceRoot 'state-replay-fake'
                    $replayArgs = @(
                        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $checkScript,
                        '-Suite', 'W05', '-Mode', 'fake', '-Database', 'postgres',
                        '-CheckIds', 'W05-FS02,W05-EN03', '-EvidenceDir', $replayEvidence, '-PythonPath', $pythonPath
                    )
                    Write-Output 'Running only Fake FS02/EN03 development replay against PostgreSQL; no paid provider calls.'
                    & $shell @replayArgs
                    $replayExitCode = $LASTEXITCODE
                    $exitCode = if ($replayExitCode -eq 0) { 0 } elseif ($replayExitCode -eq 2) { 2 } else { 1 }
                    Write-Output "Evidence root: $evidenceRoot"
                }
                elseif ($ProvenanceReplayOnly) {
                    $replayFakeEvidence = Join-Path $evidenceRoot 'provenance-replay-fake'
                    $replayRealEvidence = Join-Path $evidenceRoot 'provenance-replay-real'
                    $targetedCheckIds = 'W05-FS02,W05-EN03'
                    $replayFakeArgs = @(
                        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $checkScript,
                        '-Suite', 'W05', '-Mode', 'fake', '-Database', 'postgres',
                        '-CheckIds', $targetedCheckIds, '-EvidenceDir', $replayFakeEvidence, '-PythonPath', $pythonPath
                    )
                    Write-Output 'Running targeted Fake FS02/EN03 replay against PostgreSQL; no paid provider calls.'
                    & $shell @replayFakeArgs
                    $replayFakeExitCode = $LASTEXITCODE
                    if ($replayFakeExitCode -ne 0) {
                        Write-Output "Targeted Fake replay did not pass; Real replay was not started. Evidence root: $evidenceRoot"
                        $exitCode = if ($replayFakeExitCode -eq 2) { 2 } else { 1 }
                    }
                    else {
                        $replayRealArgs = @(
                            '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $checkScript,
                            '-Suite', 'W05', '-Mode', 'real', '-Database', 'postgres',
                            '-CheckIds', $targetedCheckIds, '-EvidenceDir', $replayRealEvidence, '-PythonPath', $pythonPath
                        )
                        Write-Output 'Running targeted Real FS02/EN03 product replay and real retrieval-to-answer integration; no full W05 suite.'
                        & $shell @replayRealArgs
                        $replayRealExitCode = $LASTEXITCODE
                        Write-Output "Evidence root: $evidenceRoot"
                        Write-Output ("Targeted replay exit codes: fake=$replayFakeExitCode real=$replayRealExitCode")
                        if ($replayRealExitCode -ne 0) {
                            $exitCode = if ($replayRealExitCode -eq 2) { 2 } else { 1 }
                        }
                    }
                }
                else {
                    $providerEvidence = Join-Path $evidenceRoot 'preflight-providers'
                    $providerCheckIds = if ($R05PreflightOnly -or $SmokeOnly) { 'W05-R05' } else { 'W05-R05,W05-EN02' }
                    $providerArgs = @(
                        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $checkScript,
                        '-Suite', 'W05', '-Mode', 'real', '-Database', 'postgres',
                        '-CheckIds', $providerCheckIds, '-EvidenceDir', $providerEvidence, '-PythonPath', $pythonPath
                    )
                    if ($R05PreflightOnly -or $SmokeOnly) {
                        Write-Output 'Running the real R05 Chat comparison only; EN02 Embedding and Rerank calls are not repeated.'
                    }
                    else {
                        Write-Output 'Running required Chat, Embedding and Rerank capability checks using development data only.'
                    }
                    & $shell @providerArgs
                    $providerExitCode = $LASTEXITCODE
                    if ($providerExitCode -ne 0) {
                        Write-Output "Provider preflight did not pass. Evidence root: $evidenceRoot"
                        $exitCode = if ($providerExitCode -eq 2) { 2 } else { 1 }
                    }
                    elseif ($SmokeOnly) {
                        # Same credentials and DB/R05 preflight as above; only the
                        # small B2a Real subset runs, never the complete suites.
                        $smokeEvidence = Join-Path $evidenceRoot 'smoke-b2a-real'
                        $smokeCheck = Join-Path $PSScriptRoot 'check_w05.py'
                        Write-Output 'Running only the W05-SMOKE-B2A Real subset (the smoke cases plus every B2a supplement case); complete W05 suites are not run.'
                        & $pythonPath $smokeCheck --check-id W05-SMOKE-B2A --mode real --evidence-dir $smokeEvidence
                        $smokeExitCode = $LASTEXITCODE
                        Write-Output "Evidence root: $evidenceRoot"
                        $exitCode = if ($smokeExitCode -eq 0) { 0 } elseif ($smokeExitCode -eq 2) { 2 } else { 1 }
                    }
                    elseif ($PreflightOnly -or $R05PreflightOnly) {
                        if ($R05PreflightOnly) {
                            Write-Output "R05 preflight passed; EN02 and complete W05 suites were not run. Evidence root: $evidenceRoot"
                        }
                        else {
                            Write-Output "Preflight passed; complete W05 suites were not run. Evidence root: $evidenceRoot"
                        }
                    }
                    else {
                        $fakeArgs = @(
                            "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $checkScript,
                            "-Suite", "W05", "-Mode", "fake", "-Database", "postgres",
                            "-EvidenceDir", $fakeEvidence, "-PythonPath", $pythonPath
                        )
                        Write-Output "Running the complete W05 Fake suite against the configured PostgreSQL fixture."
                        & $shell @fakeArgs
                        $fakeExitCode = $LASTEXITCODE

                        if ($fakeExitCode -ne 0) {
                            Write-Output "Full Fake suite did not pass; fix its recorded blockers before starting the paid real full run. Evidence root: $evidenceRoot"
                            $exitCode = if ($fakeExitCode -eq 2) { 2 } else { 1 }
                        }
                        else {
                            $realArgs = @(
                                "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $checkScript,
                                "-Suite", "W05", "-Mode", "real", "-Database", "postgres",
                                "-EvidenceDir", $realEvidence, "-PythonPath", $pythonPath
                            )
                            Write-Output "Running the complete W05 real suite; chat, embedding, and rerank capability are validated by their required calls."
                            & $shell @realArgs
                            $realExitCode = $LASTEXITCODE

                            Write-Output "Evidence root: $evidenceRoot"
                            Write-Output ("Full suite exit codes: fake=$fakeExitCode real=$realExitCode")
                            if ($realExitCode -ne 0) {
                                $exitCode = if ($realExitCode -eq 2) { 2 } else { 1 }
                            }
                        }
                    }
                }
                }
            }
        }
    }
}
catch {
    # Avoid printing exception messages that could contain a local endpoint or secret.
    Write-Output ("Local W05 run stopped after a sanitized failure (" + $_.Exception.GetType().Name + "). Credentials will be restored.")
    if ($exitCode -ne 2) { $exitCode = 1 }
}
finally {
    $sharedKey = $null
    $databasePassword = $null
    $escapedPassword = $null
    foreach ($name in $requiredNames) {
        [Environment]::SetEnvironmentVariable($name, $previousValues[$name], "Process")
    }
    Set-Location -LiteralPath $previousLocation.Path
}

exit $exitCode
