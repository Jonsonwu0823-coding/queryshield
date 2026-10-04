[CmdletBinding()]
param(
    [string]$Suite = "W01",
    [string]$Mode = "fake",
    [string]$Database = "postgres",
    [Parameter(Mandatory = $true)]
    [string]$EvidenceDir,
    [string]$PythonPath,
    [string]$CheckIds
)

$ErrorActionPreference = "Stop"
$startedAt = [DateTime]::UtcNow
$projectRoot = Split-Path -Parent $PSScriptRoot
# Windows PowerShell 5.1 has no $IsWindows, so the platform comes from .NET.
$script:isWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
$python = if ($PythonPath) {
    $PythonPath
}
elseif ($script:isWindowsHost) {
    Join-Path $projectRoot ".venv\Scripts\python.exe"
}
else {
    Join-Path (Join-Path (Join-Path $projectRoot ".venv") "bin") "python"
}
$script:checkResults = @()
$script:allRequiredCheckIds = @()
$script:requiredCheckIds = @()
$script:selectedCheckIds = @()
$script:scope = "full"
$script:databaseEvidence = "W01-DB01.txt"
$script:requiredRuntimeCheckIds = @()
$script:runtimeProfile = "not_applicable"
$script:runtimeUpstreamManifest = @()
$script:runtimeTimingScope = "not_recorded"
# These summary totals are optional until a runner has aggregated per-operation
# evidence. An uncomputed total is unknown, not zero usage.
$script:knownUsage = $null
$script:unknownUsageCount = $null
$script:requiredEngineeringCheckIds = @()
$script:capabilityManifest = @()
$script:operationUsage = @()

function Add-CheckResult {
    param(
        [string]$CheckId,
        [string]$Status,
        [string]$Command,
        [int]$ExitCode,
        [string]$EvidencePath
    )

    $script:checkResults += [ordered]@{
        check_id = $CheckId
        status = $Status
        command = $Command
        exit_code = $ExitCode
        evidence_path = $EvidencePath
    }
}

function Write-SourceManifest {
    param([string]$OutputDir)

    $excludedDirectoryNames = @(
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".vscode",
        ".idea",
        "build",
        "dist"
    )
    $excludedFilePatterns = @(
        "*.pyc",
        "*.pyo",
        "*.coverage",
        "*.code-workspace"
    )
    $roots = @(
        "src",
        "scripts",
        "migrations",
        "fixtures",
        "tests",
        "docs",
        "evals"
    )
    $manifestFiles = @()

    foreach ($root in $roots) {
        $rootPath = Join-Path $projectRoot $root
        if (-not (Test-Path -LiteralPath $rootPath -PathType Container)) {
            continue
        }

        foreach ($file in (Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force)) {
            $relativePath = $file.FullName.Substring($projectRoot.Length + 1).Replace("\", "/")
            $excluded = $false
            foreach ($part in $relativePath.Split("/")) {
                if (($excludedDirectoryNames -contains $part) -or ($part -like "*.egg-info")) {
                    $excluded = $true
                    break
                }
            }
            if ($excluded) {
                continue
            }

            foreach ($pattern in $excludedFilePatterns) {
                if ($file.Name -like $pattern) {
                    $excluded = $true
                    break
                }
            }
            if (-not $excluded) {
                $manifestFiles += $file
            }
        }
    }

    foreach ($topLevelFile in @("pyproject.toml", "README.md", "requirements.lock", ".gitignore")) {
        $topLevelPath = Join-Path $projectRoot $topLevelFile
        if (Test-Path -LiteralPath $topLevelPath -PathType Leaf) {
            $manifestFiles += Get-Item -LiteralPath $topLevelPath -Force
        }
    }

    $uniquePaths = [System.Collections.Generic.Dictionary[string, string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $manifestFiles) {
        $fullPath = [string]$file.FullName
        if (-not $uniquePaths.ContainsKey($fullPath)) {
            $uniquePaths.Add($fullPath, $fullPath)
        }
    }
    [string[]]$orderedPaths = @($uniquePaths.Values)
    [Array]::Sort($orderedPaths, [System.StringComparer]::Ordinal)

    $manifestLines = [System.Collections.Generic.List[string]]::new()
    foreach ($fullPath in $orderedPaths) {
        $relativePath = $fullPath.Substring($projectRoot.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $manifestLines.Add("$hash  $relativePath")
    }
    $manifestPath = Join-Path $OutputDir "source-manifest.txt"
    $manifestContent = [string]::Join("`n", $manifestLines.ToArray()) + "`n"
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($manifestPath, $manifestContent, $utf8WithoutBom)
    $manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

    return [ordered]@{
        path = $manifestPath
        sha256 = $manifestHash
    }
}

function Restore-MetadataToolsSetting {
    # Idempotent; a no-op before the clear in the main body has run.
    if ($null -eq $script:metadataToolsWasSet) {
        return
    }
    if ($script:metadataToolsWasSet) {
        $env:QUERYSHIELD_METADATA_TOOLS = $script:metadataToolsPrevious
    }
    else {
        Remove-Item -LiteralPath Env:QUERYSHIELD_METADATA_TOOLS -ErrorAction SilentlyContinue
    }
    $script:metadataToolsWasSet = $null
}

function Write-Summary {
    param([ValidateSet("pass", "fail", "blocked")][string]$OverallStatus)

    Restore-MetadataToolsSetting
    $manifest = Write-SourceManifest -OutputDir $EvidenceDir
    $endedAt = [DateTime]::UtcNow
    $observedIds = @($script:checkResults | ForEach-Object { $_.check_id })
    $unexecuted = @(
        $script:requiredCheckIds |
            Where-Object { $observedIds -notcontains $_ }
    )
    $summary = [ordered]@{
        contract_version = "2026-09-06.practice-v3"
        extension_version = "2026-09-09.facts-state-v1"
        runtime_extension_version = "2026-09-12.runtime-v1"
        engineering_version = "2026-09-19.engineering-v1"
        suite = $Suite
        mode = $Mode
        database = $Database
        database_evidence = $script:databaseEvidence
        mcp_transport = "not_applicable"
        fixture_version = "commerce-v1"
        scope = $script:scope
        source_manifest_sha256 = $manifest.sha256
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        all_required_check_ids = @($script:allRequiredCheckIds)
        selected_check_ids = @($script:selectedCheckIds)
        required_check_ids = @($script:requiredCheckIds)
        unexecuted_checks = $unexecuted
        checks = @($script:checkResults)
        required_runtime_checks = @($script:requiredRuntimeCheckIds)
        profile = $script:runtimeProfile
        upstream_manifest = @($script:runtimeUpstreamManifest)
        raw_evidence_paths = @(
            $script:checkResults |
                ForEach-Object { Join-Path $EvidenceDir $_.evidence_path }
        )
        timing_scope = $script:runtimeTimingScope
        known_usage = $script:knownUsage
        unknown_usage_count = $script:unknownUsageCount
        required_engineering_checks = @($script:requiredEngineeringCheckIds)
        capability_manifest = @($script:capabilityManifest)
        operation_usage = @($script:operationUsage)
        overall_status = $OverallStatus
    }
    $summaryPath = Join-Path $EvidenceDir "summary.json"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Output "summary_path=$summaryPath"
}

function Stop-Run {
    param(
        [ValidateSet("fail", "blocked")][string]$OverallStatus,
        [string]$Message,
        [int]$ExitCode
    )

    if ($Message) {
        Write-Output $Message
    }
    Write-Summary -OverallStatus $OverallStatus
    exit $ExitCode
}

function Invoke-Probe {
    param(
        [string]$CheckId,
        [string]$ScriptPath,
        [string]$EvidencePath,
        [string]$Command,
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        "blocked reason=probe_missing path=$ScriptPath" | Set-Content -LiteralPath $EvidencePath -Encoding UTF8
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Stop-Run -OverallStatus "blocked" -Message "check_blocked id=$CheckId evidence=$EvidencePath" -ExitCode 2
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeOutput = @(& $python $ScriptPath @Arguments 2>&1)
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($probeOutput.Count -eq 0) {
        $probeOutput = @("no_probe_output")
    }
    @($probeOutput | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $EvidencePath -Encoding UTF8

    if ($probeExitCode -eq 2) {
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Stop-Run -OverallStatus "blocked" -Message "check_blocked id=$CheckId evidence=$EvidencePath" -ExitCode 2
    }
    if ($probeExitCode -ne 0) {
        Add-CheckResult -CheckId $CheckId -Status "fail" -Command $Command -ExitCode $probeExitCode -EvidencePath (Split-Path -Leaf $EvidencePath)
        Stop-Run -OverallStatus "fail" -Message "check_fail id=$CheckId evidence=$EvidencePath exit_code=$probeExitCode" -ExitCode 1
    }

    Add-CheckResult -CheckId $CheckId -Status "pass" -Command $Command -ExitCode 0 -EvidencePath (Split-Path -Leaf $EvidencePath)
    Write-Output "check_pass id=$CheckId evidence=$EvidencePath"
}

function Invoke-ProbeContinue {
    param(
        [string]$CheckId,
        [string]$ScriptPath,
        [string]$EvidencePath,
        [string]$Command,
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        "blocked reason=probe_missing path=$ScriptPath" | Set-Content -LiteralPath $EvidencePath -Encoding UTF8
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_blocked id=$CheckId evidence=$EvidencePath"
        return
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeOutput = @(& $python $ScriptPath @Arguments 2>&1)
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($probeOutput.Count -eq 0) {
        $probeOutput = @("no_probe_output")
    }
    @($probeOutput | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $EvidencePath -Encoding UTF8

    if ($probeExitCode -eq 2) {
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_blocked id=$CheckId evidence=$EvidencePath"
        return
    }
    if ($probeExitCode -ne 0) {
        Add-CheckResult -CheckId $CheckId -Status "fail" -Command $Command -ExitCode $probeExitCode -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_fail id=$CheckId evidence=$EvidencePath exit_code=$probeExitCode"
        return
    }

    Add-CheckResult -CheckId $CheckId -Status "pass" -Command $Command -ExitCode 0 -EvidencePath (Split-Path -Leaf $EvidencePath)
    Write-Output "check_pass id=$CheckId evidence=$EvidencePath"
}

function Invoke-W02Process {
    param(
        [string]$CheckId,
        [string]$ScriptPath,
        [string]$EvidencePath,
        [string]$Command,
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        "blocked reason=probe_missing path=$ScriptPath" | Set-Content -LiteralPath $EvidencePath -Encoding UTF8
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_blocked id=$CheckId evidence=$EvidencePath"
        return
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeOutput = @(& $python $ScriptPath @Arguments 2>&1)
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($probeOutput.Count -eq 0) {
        $probeOutput = @("no_probe_output")
    }
    @($probeOutput | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $EvidencePath -Encoding UTF8

    if ($probeExitCode -eq 0) {
        Add-CheckResult -CheckId $CheckId -Status "pass" -Command $Command -ExitCode 0 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_pass id=$CheckId evidence=$EvidencePath"
        return
    }
    if ($probeExitCode -eq 2) {
        Add-CheckResult -CheckId $CheckId -Status "blocked" -Command $Command -ExitCode 2 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_blocked id=$CheckId evidence=$EvidencePath"
        return
    }

    Add-CheckResult -CheckId $CheckId -Status "fail" -Command $Command -ExitCode $probeExitCode -EvidencePath (Split-Path -Leaf $EvidencePath)
    Write-Output "check_fail id=$CheckId evidence=$EvidencePath exit_code=$probeExitCode"
    return
}

function Invoke-W02PythonArguments {
    param(
        [string]$CheckId,
        [string]$EvidencePath,
        [string]$Command,
        [string[]]$Arguments = @()
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeOutput = @(& $python @Arguments 2>&1)
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($probeOutput.Count -eq 0) {
        $probeOutput = @("no_probe_output")
    }
    @($probeOutput | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $EvidencePath -Encoding UTF8

    if ($probeExitCode -eq 0) {
        Add-CheckResult -CheckId $CheckId -Status "pass" -Command $Command -ExitCode 0 -EvidencePath (Split-Path -Leaf $EvidencePath)
        Write-Output "check_pass id=$CheckId evidence=$EvidencePath"
        return
    }
    Add-CheckResult -CheckId $CheckId -Status "fail" -Command $Command -ExitCode $probeExitCode -EvidencePath (Split-Path -Leaf $EvidencePath)
    Write-Output "check_fail id=$CheckId evidence=$EvidencePath exit_code=$probeExitCode"
    return
}

function Invoke-W02Suite {
    $offlineEvidence = Join-Path $EvidenceDir "W02-OFFLINE.txt"
    $offlineBaseTemp = Join-Path $EvidenceDir "pytest-tmp"
    $offlineArguments = @(
        "-m", "pytest",
        "tests/test_model_adapters.py",
        "tests/test_query_proposals.py",
        "tests/test_durable_call_store.py",
        "tests/test_api_call_identity.py",
        "tests/test_sql_policy.py",
        "tests/test_guarded_query.py",
        "tests/test_w02_t05_probe.py",
        "tests/test_w02_t06_probe.py",
        "tests/test_w02_fs_probe.py",
        "-q",
        "-p", "no:cacheprovider",
        "--basetemp", $offlineBaseTemp
    )
    $requestedProviderMode = $env:QUERYSHIELD_PROVIDER_MODE
    $env:QUERYSHIELD_PROVIDER_MODE = "fake"
    Invoke-W02PythonArguments `
        -CheckId "W02-OFFLINE" `
        -EvidencePath $offlineEvidence `
        -Command "python -m pytest tests/test_model_adapters.py tests/test_query_proposals.py tests/test_durable_call_store.py tests/test_api_call_identity.py tests/test_sql_policy.py tests/test_guarded_query.py tests/test_w02_t05_probe.py tests/test_w02_t06_probe.py tests/test_w02_fs_probe.py -q -p no:cacheprovider --basetemp $offlineBaseTemp" `
        -Arguments $offlineArguments
    $env:QUERYSHIELD_PROVIDER_MODE = $requestedProviderMode

    $providerEvidence = Join-Path $EvidenceDir "W02-PROVIDER.txt"
    $providerJson = Join-Path $EvidenceDir "W02-PROVIDER.json"
    Invoke-W02Process `
        -CheckId "W02-PROVIDER" `
        -ScriptPath (Join-Path $PSScriptRoot "model_probe.py") `
        -EvidencePath $providerEvidence `
        -Command "python scripts/model_probe.py -Mode $Mode -Output W02-PROVIDER.json" `
        -Arguments @("-Mode", $Mode, "-Output", $providerJson)

    $databaseEvidencePath = Join-Path $EvidenceDir "W02-DB.txt"
    if (-not $env:QUERYSHIELD_DATABASE_URL) {
        "blocked reason=QUERYSHIELD_DATABASE_URL_missing" | Set-Content -LiteralPath $databaseEvidencePath -Encoding UTF8
        Add-CheckResult -CheckId "W02-DB" -Status "blocked" -Command "python scripts/check_db.py" -ExitCode 2 -EvidencePath "W02-DB.txt"
        Write-Output "check_blocked id=W02-DB evidence=$databaseEvidencePath"
    }
    else {
        Invoke-W02Process `
            -CheckId "W02-DB" `
            -ScriptPath (Join-Path $PSScriptRoot "check_db.py") `
            -EvidencePath $databaseEvidencePath `
            -Command "python scripts/check_db.py"
    }

    if ($Mode -eq "real") {
        # Retired in B2b: it exercised the deleted W02 single-pass /queries path
        # (fixed-question prompt and row-count answers). The real model -> guarded
        # SQL -> tenant scope chain over HTTP is covered together by the B2b HTTP Real
        # smoke (scripts/b2b-local-http-smoke.ps1) and the W05 full Real run
        # (scripts/w05-local-real.ps1). The full Real run was blocked by W05-X01 after
        # B2b; B3a made X01 read this repository, so it starts again once the complete
        # W05 Fake suite passes on the machine.
        $realChainEvidence = Join-Path $EvidenceDir "W02-REAL-CHAIN.txt"
        "not_applicable reason=retired_by_B2b replacement=scripts/b2b-local-http-smoke.ps1+scripts/w05-local-real.ps1(W05-full-real)" | Set-Content -LiteralPath $realChainEvidence -Encoding UTF8
        Add-CheckResult -CheckId "W02-REAL-CHAIN" -Status "not_applicable" -Command "retired: python scripts/w02_real_chain_probe.py" -ExitCode 0 -EvidencePath "W02-REAL-CHAIN.txt"
        Write-Output "check_not_applicable id=W02-REAL-CHAIN evidence=$realChainEvidence"
    }

    $factsEvidence = Join-Path $EvidenceDir "W02-FS01.txt"
    Invoke-W02Process `
        -CheckId "W02-FS01" `
        -ScriptPath (Join-Path $PSScriptRoot "w02_fs_probe.py") `
        -EvidencePath $factsEvidence `
        -Command "python scripts/w02_fs_probe.py -CheckId W02-FS01 -EvidenceDir $EvidenceDir" `
        -Arguments @("-CheckId", "W02-FS01", "-EvidenceDir", $EvidenceDir)

    $factsCheckEvidence = Join-Path $EvidenceDir "W02-FS02.txt"
    Invoke-W02Process `
        -CheckId "W02-FS02" `
        -ScriptPath (Join-Path $PSScriptRoot "w02_fs_probe.py") `
        -EvidencePath $factsCheckEvidence `
        -Command "python scripts/w02_fs_probe.py -CheckId W02-FS02 -EvidenceDir $EvidenceDir" `
        -Arguments @("-CheckId", "W02-FS02", "-EvidenceDir", $EvidenceDir)

    $statuses = @($script:checkResults | ForEach-Object { $_.status })
    if ($statuses -contains "fail") {
        Write-Summary -OverallStatus "fail"
        exit 1
    }
    if ($statuses -contains "blocked" -or $statuses -contains "not_run") {
        Write-Summary -OverallStatus "blocked"
        exit 2
    }
    Write-Summary -OverallStatus "pass"
    exit 0
}

function Invoke-W03Suite {
    $script:requiredRuntimeCheckIds = @("W03-RT01", "W03-RT02", "W03-RT03")
    $script:requiredEngineeringCheckIds = @("W03-EN01", "W03-EN02", "W03-EN03", "W03-EN04")
    $script:capabilityManifest = @("embedding-v1", "vector-index-v1", "hybrid-v1", "retrieval-evidence-v1", "run-config-v1", "qs-action-schema-v1", "bounded-langgraph-v1", "qs-parallel-v1", "shared-parallel-budget-v1", "verified-facts-v1", "agent-trace-v1", "t06-behavior-matrix-v1")
    $script:operationUsage = @(
        "W03-EN02 raw evidence records one OperationUsage per embedding call",
        "W03-EN03 raw evidence records query embedding call/return and selected sources",
        "W03-EN04 run records fixed prompt/schema/catalog/snapshot versions",
        "W03-RT02 fake parallel branches preserve branch identity and peak_active<=2",
        "W03-RT03 fake parallel branches share the run tool budget and expose PENDING branches",
        "W03-T05 model/tool trace is redacted and usage is aggregated once per run",
        "W03-T06 fake matrix records five bounded-agent behaviors and real mode is separate",
        "W03-DB01 runs the real PostgreSQL read-only engine check and preserves blocked/fail distinction",
        "W03-FS01 runs the formal server-facts positive/empty/change cases with known usage",
        "W03-FS02 runs formal evidence forgery rejection cases without repair or public facts"
    )
    $script:runtimeProfile = if ($Mode -eq "fake") { "fake-context-budget-v1" } else { "not_applicable" }
    $script:runtimeUpstreamManifest = @(
        "src/queryshield/tools/semantic.py",
        "src/queryshield/facts/facts.py",
        "src/queryshield/agent/proposals.py",
        "src/queryshield/agent/parallel.py"
    )
    $script:runtimeTimingScope = if ($Mode -eq "fake") {
        "deterministic in-process; no model/provider/database call"
    }
    else {
        "not_applicable: W03-RT01 is fake-only"
    }

    if ($script:selectedCheckIds -contains "W03-EN01") {
        if ($Mode -ne "fake") {
            $notApplicableEvidence = Join-Path $EvidenceDir "W03-EN01.txt"
            "not_applicable reason=W03-EN01_requires_fake_mode" | Set-Content -LiteralPath $notApplicableEvidence -Encoding UTF8
            Add-CheckResult -CheckId "W03-EN01" -Status "not_applicable" -Command "python scripts/check_w03_en01.py" -ExitCode 0 -EvidencePath "W03-EN01.txt"
        }
        else {
            $knowledgeEvidence = Join-Path $EvidenceDir "W03-EN01.txt"
            $sourceRoot = Join-Path (Join-Path $projectRoot "fixtures") "knowledge"
            $registryPath = Join-Path $sourceRoot "source_registry.json"
            $scriptPath = Join-Path $PSScriptRoot "check_w03_en01.py"
            Invoke-ProbeContinue `
                -CheckId "W03-EN01" `
                -ScriptPath $scriptPath `
                -EvidencePath $knowledgeEvidence `
                -Command "python scripts/check_w03_en01.py --source-root fixtures/knowledge --registry fixtures/knowledge/source_registry.json --catalog-version catalog-v2" `
                -Arguments @(
                    "--source-root", $sourceRoot,
                    "--registry", $registryPath,
                    "--catalog-version", "catalog-v2"
                )
        }
    }

    if ($script:selectedCheckIds -contains "W03-RT01") {
        if ($Mode -ne "fake") {
            $runtimeEvidence = Join-Path $EvidenceDir "W03-RT01.txt"
            "not_applicable reason=W03-RT01_requires_fake_mode" | Set-Content -LiteralPath $runtimeEvidence -Encoding UTF8
            Add-CheckResult -CheckId "W03-RT01" -Status "not_applicable" -Command "python scripts/check_w03_rt01.py" -ExitCode 0 -EvidencePath "W03-RT01.txt"
        }
        else {
            $runtimeEvidence = Join-Path $EvidenceDir "W03-RT01.txt"
            Invoke-ProbeContinue `
                -CheckId "W03-RT01" `
                -ScriptPath (Join-Path $PSScriptRoot "check_w03_rt01.py") `
                -EvidencePath $runtimeEvidence `
                -Command "python scripts/check_w03_rt01.py"
        }
    }

    if ($script:selectedCheckIds -contains "W03-RT02") {
        if ($Mode -ne "fake") {
            $parallelEvidence = Join-Path $EvidenceDir "W03-RT02.txt"
            "not_applicable reason=W03-RT02_requires_fake_mode" | Set-Content -LiteralPath $parallelEvidence -Encoding UTF8
            Add-CheckResult -CheckId "W03-RT02" -Status "not_applicable" -Command "python scripts/check_w03_rt02.py" -ExitCode 0 -EvidencePath "W03-RT02.txt"
        }
        else {
            $parallelEvidence = Join-Path $EvidenceDir "W03-RT02.txt"
            Invoke-ProbeContinue `
                -CheckId "W03-RT02" `
                -ScriptPath (Join-Path $PSScriptRoot "check_w03_rt02.py") `
                -EvidencePath $parallelEvidence `
                -Command "python scripts/check_w03_rt02.py --output-dir $EvidenceDir" `
                -Arguments @(
                    "--output-dir", $EvidenceDir
                )
        }
    }

    if ($script:selectedCheckIds -contains "W03-RT03") {
        if ($Mode -ne "fake") {
            $budgetEvidence = Join-Path $EvidenceDir "W03-RT03.txt"
            "not_applicable reason=W03-RT03_requires_fake_mode" | Set-Content -LiteralPath $budgetEvidence -Encoding UTF8
            Add-CheckResult -CheckId "W03-RT03" -Status "not_applicable" -Command "python scripts/check_w03_rt03.py" -ExitCode 0 -EvidencePath "W03-RT03.txt"
        }
        else {
            $budgetEvidence = Join-Path $EvidenceDir "W03-RT03.txt"
            Invoke-ProbeContinue `
                -CheckId "W03-RT03" `
                -ScriptPath (Join-Path $PSScriptRoot "check_w03_rt03.py") `
                -EvidencePath $budgetEvidence `
                -Command "python scripts/check_w03_rt03.py --output-dir $EvidenceDir" `
                -Arguments @(
                    "--output-dir", $EvidenceDir
                )
        }
    }

    $snapshotPath = Join-Path (Join-Path (Join-Path (Join-Path $projectRoot "fixtures") "knowledge") "snapshots") "knowledge-v1-9f580dd7f887ed0a.json"

    if ($script:selectedCheckIds -contains "W03-EN02") {
        if ($Mode -eq "fake") {
            $script:runtimeProfile = "fake-embedding-index-v1"
        }
        else {
            $script:runtimeProfile = "real-openai-compatible-embedding-v1"
        }
        $embeddingEvidence = Join-Path $EvidenceDir "W03-EN02.txt"
        Invoke-ProbeContinue `
            -CheckId "W03-EN02" `
            -ScriptPath (Join-Path $PSScriptRoot "check_w03_en02.py") `
            -EvidencePath $embeddingEvidence `
            -Command "python scripts/check_w03_en02.py --mode $Mode --snapshot fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json --output-dir $EvidenceDir" `
            -Arguments @(
                "--mode", $Mode,
                "--snapshot", $snapshotPath,
                "--output-dir", $EvidenceDir
            )
    }

    if ($script:selectedCheckIds -contains "W03-EN03") {
        if ($Mode -eq "fake") {
            $script:runtimeProfile = "fake-hybrid-v1"
        }
        else {
            $script:runtimeProfile = "real-hybrid-v1"
        }
        $hybridEvidence = Join-Path $EvidenceDir "W03-EN03.txt"
        Invoke-ProbeContinue `
            -CheckId "W03-EN03" `
            -ScriptPath (Join-Path $PSScriptRoot "check_w03_en03.py") `
            -EvidencePath $hybridEvidence `
            -Command "python scripts/check_w03_en03.py --mode $Mode --snapshot fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json --output-dir $EvidenceDir" `
            -Arguments @(
                "--mode", $Mode,
                "--snapshot", $snapshotPath,
                "--output-dir", $EvidenceDir
            )
    }

    if ($script:selectedCheckIds -contains "W03-EN04") {
        if ($Mode -ne "fake") {
            $versionEvidence = Join-Path $EvidenceDir "W03-EN04.txt"
            "not_applicable reason=W03-EN04_requires_fake_mode" | Set-Content -LiteralPath $versionEvidence -Encoding UTF8
            Add-CheckResult -CheckId "W03-EN04" -Status "not_applicable" -Command "python scripts/check_w03_en04.py --mode fake" -ExitCode 0 -EvidencePath "W03-EN04.txt"
        }
        else {
            $versionEvidence = Join-Path $EvidenceDir "W03-EN04.txt"
            Invoke-ProbeContinue `
                -CheckId "W03-EN04" `
                -ScriptPath (Join-Path $PSScriptRoot "check_w03_en04.py") `
                -EvidencePath $versionEvidence `
                -Command "python scripts/check_w03_en04.py --mode fake --snapshot fixtures/knowledge/snapshots/knowledge-v1-9f580dd7f887ed0a.json --output-dir $EvidenceDir" `
                -Arguments @(
                    "--mode", "fake",
                    "--snapshot", $snapshotPath,
                    "--output-dir", $EvidenceDir
                )
        }
    }

    foreach ($ticketCheck in @("W03-T03", "W03-T04", "W03-T05", "W03-T06")) {
        if ($script:selectedCheckIds -notcontains $ticketCheck) {
            continue
        }

        $ticketEvidence = Join-Path $EvidenceDir "$ticketCheck.txt"
        $ticketFile = switch ($ticketCheck) {
            "W03-T03" { "check_w03_t03.py" }
            "W03-T04" { "check_w03_t04.py" }
            "W03-T05" { "check_w03_t05.py" }
            "W03-T06" { "check_w03_t06.py" }
        }
        if ($Mode -eq "real" -and $ticketCheck -ne "W03-T06") {
            "not_applicable reason=${ticketCheck}_fake_behavior_matrix_is_recorded_in_fake_mode" | Set-Content -LiteralPath $ticketEvidence -Encoding UTF8
            Add-CheckResult -CheckId $ticketCheck -Status "not_applicable" -Command "python scripts/$ticketFile --output-dir $EvidenceDir" -ExitCode 0 -EvidencePath (Split-Path -Leaf $ticketEvidence)
            continue
        }

        $ticketScript = Join-Path $PSScriptRoot $ticketFile
        $ticketArguments = if ($ticketCheck -eq "W03-T06") {
            @("--mode", $Mode, "--output-dir", $EvidenceDir)
        }
        else {
            @("--output-dir", $EvidenceDir)
        }
        Invoke-ProbeContinue `
            -CheckId $ticketCheck `
            -ScriptPath $ticketScript `
            -EvidencePath $ticketEvidence `
            -Command "python scripts/$($ticketScript.Substring($PSScriptRoot.Length + 1)) $($ticketArguments -join ' ')" `
            -Arguments $ticketArguments
    }

    $databaseEvidence = Join-Path $EvidenceDir "W03-DB01.txt"
    Invoke-ProbeContinue `
        -CheckId "W03-DB01" `
        -ScriptPath (Join-Path $PSScriptRoot "check_db.py") `
        -EvidencePath $databaseEvidence `
        -Command "python scripts/check_db.py" 

    foreach ($factsCheck in @("W03-FS01", "W03-FS02")) {
        $factsEvidence = Join-Path $EvidenceDir "$factsCheck.txt"
        if ($Mode -eq "real") {
            "not_applicable reason=${factsCheck}_uses_fake_server_facts_boundary; real_database_and_model_path_is_W03-DB01_and_W03-T06" | Set-Content -LiteralPath $factsEvidence -Encoding UTF8
            Add-CheckResult -CheckId $factsCheck -Status "not_applicable" -Command "python scripts/check_w03_fs.py --check-id $factsCheck --mode fake --output-dir $EvidenceDir" -ExitCode 0 -EvidencePath (Split-Path -Leaf $factsEvidence)
            continue
        }

        Invoke-ProbeContinue `
            -CheckId $factsCheck `
            -ScriptPath (Join-Path $PSScriptRoot "check_w03_fs.py") `
            -EvidencePath $factsEvidence `
            -Command "python scripts/check_w03_fs.py --check-id $factsCheck --mode fake --output-dir $EvidenceDir" `
            -Arguments @(
                "--check-id", $factsCheck,
                "--mode", "fake",
                "--output-dir", $EvidenceDir
            )
    }

    $statuses = @($script:checkResults | ForEach-Object { $_.status })
    if ($statuses -contains "fail") {
        Write-Summary -OverallStatus "fail"
        exit 1
    }
    if ($statuses -contains "blocked" -or $statuses -contains "not_run") {
        Write-Summary -OverallStatus "blocked"
        exit 2
    }
    Write-Summary -OverallStatus "pass"
    exit 0
}

function Invoke-W04Suite {
    $script:requiredRuntimeCheckIds = @("W04-RT01", "W04-RT02")
    $script:requiredEngineeringCheckIds = @("W04-EN01", "W04-EN02", "W04-EN03", "W04-EN04", "W04-EN05")
    $script:capabilityManifest = @(
        "w04-identity-rbac-v1",
        "w04-postgres-rls-v1",
        "w04-approval-recovery-v1",
        "w04-knowledge-snapshot-acl-v1",
        "w04-confirmed-preferences-v1",
        "w04-context-restore-v1",
        "w04-parallel-durable-v1",
        "w04-events-sse-v1"
    )
    $script:operationUsage = @(
        "W04-R01/R02/W04-DB01 use the real queryshield_ro PostgreSQL role; missing database is blocked",
        "W04-R03/R04/R05/R06/R07/R08 use deterministic Fake/model and local durable state",
        "W04-FS01/FS02 preserve approval clock, action digest, restart and replay boundaries",
        "W04-RT01/RT02 verify bounded parallel branches, cancellation and active-run capacity",
        "W04-EN04 starts a separate local uvicorn HTTP/SSE process and checks persistent replay",
        "W04-X03 records implementation, learner, Fake and real-database attribution separately"
    )
    $script:runtimeProfile = if ($Mode -eq "fake") { "fake-model-real-postgres-w04-v1" } else { "real-mode-db-only-w04-v1" }
    $script:runtimeUpstreamManifest = @(
        "src/queryshield/auth/identity.py",
        "src/queryshield/db/guarded.py",
        "src/queryshield/db/w04_state.py",
        "src/queryshield/approval/service.py",
        "src/queryshield/agent/parallel_durable.py",
        "src/queryshield/knowledge/snapshots.py",
        "src/queryshield/agent/context_runtime.py",
        "src/queryshield/api/main.py"
    )
    $script:runtimeTimingScope = "deterministic Fake checks plus real local HTTP; PostgreSQL checks are separately reported"

    $fakeOnlyIds = @(
        "W04-R03", "W04-R04", "W04-R05", "W04-R06", "W04-R07", "W04-R08",
        "W04-FS01", "W04-FS02", "W04-RT01", "W04-RT02",
        "W04-EN01", "W04-EN02", "W04-EN03", "W04-EN04", "W04-EN05"
    )
    foreach ($checkId in $script:selectedCheckIds) {
        $evidencePath = Join-Path $EvidenceDir "$checkId.txt"
        if ($Mode -eq "real" -and $fakeOnlyIds -contains $checkId) {
            "not_applicable reason=$checkId_requires_W04_fake_model_boundary" | Set-Content -LiteralPath $evidencePath -Encoding UTF8
            Add-CheckResult -CheckId $checkId -Status "not_applicable" -Command "python scripts/check_w04.py --check-id $checkId --mode real --output-dir $EvidenceDir" -ExitCode 0 -EvidencePath (Split-Path -Leaf $evidencePath)
            continue
        }
        Invoke-ProbeContinue `
            -CheckId $checkId `
            -ScriptPath (Join-Path $PSScriptRoot "check_w04.py") `
            -EvidencePath $evidencePath `
            -Command "python scripts/check_w04.py --check-id $checkId --mode $Mode --output-dir $EvidenceDir" `
            -Arguments @(
                "--check-id", $checkId,
                "--mode", $Mode,
                "--output-dir", $EvidenceDir
            )
    }

    $statuses = @($script:checkResults | ForEach-Object { $_.status })
    if ($statuses -contains "fail") {
        Write-Summary -OverallStatus "fail"
        exit 1
    }
    if ($statuses -contains "blocked" -or $statuses -contains "not_run") {
        Write-Summary -OverallStatus "blocked"
        exit 2
    }
    Write-Summary -OverallStatus "pass"
    exit 0
}

function Invoke-W05Suite {
    $script:requiredRuntimeCheckIds = @("W05-RT01", "W05-RT02", "W05-RT03")
    $script:requiredEngineeringCheckIds = @("W05-EN01", "W05-EN02", "W05-EN03")
    $script:capabilityManifest = @(
        "w05-state-case-loader-oracle-controls-v1",
        "w05-versioned-retrieval-development-eval-v1",
        "w05-b0-b1-shared-boundary-regression-controls-v1",
        "w05-rerank-adapter-and-fake-protocol-controls-v1",
        "w05-empty-run-report-contract-controls-v1"
    )
    $script:operationUsage = @(
        "W05-R01/R06 verify frozen dataset and sealed-family commitments without opening holdout before T05",
        "W05-R02 compares B0/B1 security and shared-configuration fingerprints; real fingerprinting is blocked when model configuration is missing",
        "W05-R03/R04 exercise denominator, unknown-usage, semantic-fact and security negative controls; they do not represent full-dataset product outcomes",
        "W05-R05 fake mode records a two-scenario regression; real mode requires one successful and one failed real replay",
        "W05-FS01 checks state-case quotas, C10 pairs and oracle-order controls; product stateful replay is not_run",
        "W05-RT01/02/03 run parallel harness controls, a serial/parallel Fake pilot and development retrieval/context; PostgreSQL remains required where stated",
        "W05-EN01 evaluates development retrieval; W05-EN02 has Fake protocol controls and a real configured path; W05-EN03 full end-to-end replay/report remains blocked",
        "W05-DB01 always requires the real queryshield_ro PostgreSQL fixture in both provider modes",
        "Unknown and failed provider calls remain in per-case denominators and unknown usage remains null"
    )
    $script:runtimeProfile = if ($Mode -eq "fake") { "fake-model-real-postgres-w05-v1" } else { "real-model-real-postgres-w05-v1" }
    $script:runtimeUpstreamManifest = @(
        "src/queryshield/evaluation/",
        "src/queryshield/knowledge/retrieval.py",
        "src/queryshield/providers/rerank.py",
        "evals/w05/",
        "fixtures/knowledge/source_registry.json",
        "fixtures/semantic/catalog-v2.json",
        "fixtures/commerce-v1.md",
        "migrations/001_commerce_v1.sql"
    )
    $script:runtimeTimingScope = "per-case retrieval/model calls and SQL execution; embedding index-build calls reported separately"

    $fakeOnlyIds = @("W05-FS01", "W05-RT01", "W05-RT02", "W05-RT03", "W05-EN01")
    foreach ($checkId in $script:selectedCheckIds) {
        $evidencePath = Join-Path $EvidenceDir "$checkId.txt"
        if ($Mode -eq "real" -and $fakeOnlyIds -contains $checkId) {
            "not_applicable reason=$checkId_requires_W05_fake_harness" | Set-Content -LiteralPath $evidencePath -Encoding UTF8
            Add-CheckResult -CheckId $checkId -Status "not_applicable" -Command "python scripts/check_w05.py --check-id $checkId --mode real --evidence-dir $EvidenceDir" -ExitCode 0 -EvidencePath (Split-Path -Leaf $evidencePath)
            continue
        }
        Invoke-ProbeContinue `
            -CheckId $checkId `
            -ScriptPath (Join-Path $PSScriptRoot "check_w05.py") `
            -EvidencePath $evidencePath `
            -Command "python scripts/check_w05.py --check-id $checkId --mode $Mode --evidence-dir $EvidenceDir" `
            -Arguments @(
                "--check-id", $checkId,
                "--mode", $Mode,
                "--evidence-dir", $EvidenceDir
            )
    }

    $statuses = @($script:checkResults | ForEach-Object { $_.status })
    if ($statuses -contains "fail") {
        Write-Summary -OverallStatus "fail"
        exit 1
    }
    if ($statuses -contains "blocked" -or $statuses -contains "not_run") {
        Write-Summary -OverallStatus "blocked"
        exit 2
    }
    Write-Summary -OverallStatus "pass"
    exit 0
}

function Invoke-RuntimePreflight {
    $preflightScript = Join-Path $PSScriptRoot "check_runtime.py"
    $evidencePath = Join-Path $EvidenceDir "runtime-preflight.txt"
    $command = "python scripts/check_runtime.py"

    if (-not (Test-Path -LiteralPath $preflightScript -PathType Leaf)) {
        "blocked reason=runtime_preflight_missing path=$preflightScript" | Set-Content -LiteralPath $evidencePath -Encoding UTF8
        Add-CheckResult -CheckId "CHECK-RUNTIME" -Status "blocked" -Command $command -ExitCode 2 -EvidencePath (Split-Path -Leaf $evidencePath)
        Stop-Run -OverallStatus "blocked" -Message "check_blocked id=CHECK-RUNTIME evidence=$evidencePath" -ExitCode 2
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $preflightOutput = @(& $python $preflightScript 2>&1)
    $preflightExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($preflightOutput.Count -eq 0) {
        $preflightOutput = @("no_runtime_preflight_output")
    }
    @($preflightOutput | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $evidencePath -Encoding UTF8

    if ($preflightExitCode -eq 2) {
        Add-CheckResult -CheckId "CHECK-RUNTIME" -Status "blocked" -Command $command -ExitCode 2 -EvidencePath (Split-Path -Leaf $evidencePath)
        Stop-Run -OverallStatus "blocked" -Message "check_blocked id=CHECK-RUNTIME evidence=$evidencePath" -ExitCode 2
    }
    if ($preflightExitCode -ne 0) {
        Add-CheckResult -CheckId "CHECK-RUNTIME" -Status "fail" -Command $command -ExitCode $preflightExitCode -EvidencePath (Split-Path -Leaf $evidencePath)
        Stop-Run -OverallStatus "fail" -Message "check_fail id=CHECK-RUNTIME evidence=$evidencePath exit_code=$preflightExitCode" -ExitCode 1
    }

    Add-CheckResult -CheckId "CHECK-RUNTIME" -Status "pass" -Command $command -ExitCode 0 -EvidencePath (Split-Path -Leaf $evidencePath)
    Write-Output "check_pass id=CHECK-RUNTIME evidence=$evidencePath"
}

$script:allRequiredCheckIds = switch ($Suite) {
    "DB-SMOKE" { @("DB-SMOKE") }
    "W03" { @("W03-EN01", "W03-RT01", "W03-RT02", "W03-RT03", "W03-EN02", "W03-EN03", "W03-EN04", "W03-T03", "W03-T04", "W03-T05", "W03-T06", "W03-DB01", "W03-FS01", "W03-FS02") }
    "W04" { @("W04-R01", "W04-R02", "W04-R03", "W04-R04", "W04-R05", "W04-R06", "W04-R07", "W04-R08", "W04-X01", "W04-X02", "W04-X03", "W04-DB01", "W04-FS01", "W04-FS02", "W04-RT01", "W04-RT02", "W04-EN01", "W04-EN02", "W04-EN03", "W04-EN04", "W04-EN05") }
    "W05" { @("W05-R01", "W05-R02", "W05-R03", "W05-R04", "W05-R05", "W05-R06", "W05-R07", "W05-X01", "W05-X02", "W05-X03", "W05-DB01", "W05-FS01", "W05-FS02", "W05-RT01", "W05-RT02", "W05-RT03", "W05-EN01", "W05-EN02", "W05-EN03") }
    "W02" {
        if ($Mode -eq "real") {
            @("W02-OFFLINE", "W02-PROVIDER", "W02-DB", "W02-REAL-CHAIN", "W02-FS01", "W02-FS02")
        }
        else {
            @("W02-OFFLINE", "W02-PROVIDER", "W02-DB", "W02-FS01", "W02-FS02")
        }
    }
    default { @("W01-DB01", "W01-API", "W01-FS01", "W01-T05-FAULTS", "W01-FS02", "W01-B06-IDENTITY") }
}
$script:requiredCheckIds = @($script:allRequiredCheckIds)
$script:selectedCheckIds = @($script:allRequiredCheckIds)
$script:scope = "full"
$validationErrors = @()
if (-not [string]::IsNullOrWhiteSpace($CheckIds)) {
    $script:scope = "partial"
    $requestedCheckIds = @(
        $CheckIds -split "," |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    if ($Suite -notin @("W03", "W04", "W05")) {
        $validationErrors = @("CheckIds_is_registered_for_W03_W04_and_W05_only")
    }
    elseif ($requestedCheckIds.Count -eq 0) {
        $validationErrors = @("CheckIds_empty")
    }
    else {
        $unknownCheckIds = @(
            $requestedCheckIds |
                Where-Object { $script:allRequiredCheckIds -notcontains $_ }
        )
        $duplicateCheckIds = @(
            $requestedCheckIds |
                Group-Object |
                Where-Object { $_.Count -gt 1 } |
                ForEach-Object { $_.Name }
        )
        if ($unknownCheckIds.Count -gt 0) {
            $validationErrors = @("unknown_CheckIds=$($unknownCheckIds -join ',')")
        }
        elseif ($duplicateCheckIds.Count -gt 0) {
            $validationErrors = @("duplicate_CheckIds=$($duplicateCheckIds -join ',')")
        }
        else {
            $script:requiredCheckIds = @($requestedCheckIds)
            $script:selectedCheckIds = @($requestedCheckIds)
        }
    }
}
$script:databaseEvidence = switch ($Suite) {
    "DB-SMOKE" { "DB-SMOKE.txt" }
    "W02" { "W02-DB.txt" }
    "W03" { "W03-DB01.txt" }
    "W04" { "W04-DB01.txt" }
    "W05" { "W05-DB01.txt" }
    default { "W01-DB01.txt" }
}

$evidenceDirIsAbsolute = if ($script:isWindowsHost) {
    $EvidenceDir -match '^[A-Za-z]:[\\/]' -or $EvidenceDir -match '^\\\\'
}
else {
    $EvidenceDir -match '^/'
}
if (-not $evidenceDirIsAbsolute) {
    Write-Output "check_blocked reason=evidence_dir_must_be_absolute"
    exit 2
}

New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null

if ($Suite -notin @("W01", "W02", "W03", "W04", "W05", "DB-SMOKE")) {
    $validationErrors += "unsupported_suite"
}
if ($Suite -eq "W05" -and -not $PSBoundParameters.ContainsKey("Mode")) {
    $validationErrors += "W05_Mode_must_be_explicit"
}
if ($Mode -notin @("fake", "real")) {
    $validationErrors += "unsupported_mode"
}
if ($Database -ne "postgres") {
    $validationErrors += "unsupported_database"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $validationErrors += "python_missing"
}
if ($Suite -notin @("W02", "W03", "W04", "W05") -and -not $env:QUERYSHIELD_DATABASE_URL) {
    $validationErrors += "QUERYSHIELD_DATABASE_URL_missing"
}

if ($validationErrors.Count -gt 0) {
    $blockedEvidence = Join-Path $EvidenceDir "check-blocked.txt"
    $validationErrors | ForEach-Object { "blocked reason=$($_)" } | Set-Content -LiteralPath $blockedEvidence -Encoding UTF8
    Add-CheckResult -CheckId "CHECK-INPUT" -Status "blocked" -Command "check.ps1 input validation" -ExitCode 2 -EvidencePath "check-blocked.txt"
    Stop-Run -OverallStatus "blocked" -Message "check_blocked reason=$($validationErrors -join ',') evidence=$blockedEvidence" -ExitCode 2
}

# M11: with QUERYSHIELD_METADATA_TOOLS set, W04-EN04 fails (the SSE stream gains a
# metadata_session event) and the checks are meant to exercise the default local
# metadata tools.  Clear it for the run; Write-Summary restores the caller's value.
$script:metadataToolsWasSet = Test-Path -LiteralPath Env:QUERYSHIELD_METADATA_TOOLS
$script:metadataToolsPrevious = $env:QUERYSHIELD_METADATA_TOOLS
Remove-Item -LiteralPath Env:QUERYSHIELD_METADATA_TOOLS -ErrorAction SilentlyContinue
$env:QUERYSHIELD_PROVIDER_MODE = $Mode

if ($Suite -eq "W04") {
    Invoke-W04Suite
}

if ($Suite -eq "W05") {
    Invoke-W05Suite
}

if ($Suite -eq "W03") {
    Invoke-W03Suite
}

if ($Suite -eq "W02") {
    Invoke-W02Suite
}

if ($Suite -eq "DB-SMOKE") {
    Invoke-RuntimePreflight
    $smokeEvidence = Join-Path $EvidenceDir "DB-SMOKE.txt"
    Invoke-Probe -CheckId "DB-SMOKE" -ScriptPath (Join-Path $PSScriptRoot "check_db_smoke.py") -EvidencePath $smokeEvidence -Command "python scripts/check_db_smoke.py"
    Write-Summary -OverallStatus "pass"
    exit 0
}

$identityEvidence = Join-Path $EvidenceDir "W01-B06-IDENTITY.txt"
Invoke-Probe -CheckId "W01-B06-IDENTITY" -ScriptPath (Join-Path $PSScriptRoot "check_identity.py") -EvidencePath $identityEvidence -Command "python scripts/check_identity.py"
Invoke-RuntimePreflight

if ($Mode -eq "real") {
    $notRunChecks = @(
        @{ id = "W01-DB01"; command = "python scripts/check_db.py" },
        @{ id = "W01-FS01"; command = "python scripts/check_commerce.py" },
        @{ id = "W01-T05-FAULTS"; command = "python scripts/check_faults.py" },
        @{ id = "W01-FS02"; command = "python scripts/check_fs02.py" }
    )
    foreach ($item in $notRunChecks) {
        $evidencePath = Join-Path $EvidenceDir ($item.id + ".txt")
        "not_run reason=real_provider_not_implemented" | Set-Content -LiteralPath $evidencePath -Encoding UTF8
        Add-CheckResult -CheckId $item.id -Status "not_run" -Command $item.command -ExitCode 2 -EvidencePath (Split-Path -Leaf $evidencePath)
    }
    $apiEvidencePath = Join-Path $EvidenceDir "W01-API.txt"
    "blocked reason=real_provider_not_implemented" | Set-Content -LiteralPath $apiEvidencePath -Encoding UTF8
    Add-CheckResult -CheckId "W01-API" -Status "blocked" -Command "python scripts/check_api.py --mode real" -ExitCode 2 -EvidencePath "W01-API.txt"
    Stop-Run -OverallStatus "blocked" -Message "check_blocked reason=real_provider_not_implemented" -ExitCode 2
}

$dbEvidence = Join-Path $EvidenceDir "W01-DB01.txt"
$apiEvidence = Join-Path $EvidenceDir "W01-API.txt"
$fs01Evidence = Join-Path $EvidenceDir "W01-FS01.txt"
$faultEvidence = Join-Path $EvidenceDir "W01-T05-FAULTS.txt"
$fs02Evidence = Join-Path $EvidenceDir "W01-FS02.txt"

Invoke-Probe -CheckId "W01-DB01" -ScriptPath (Join-Path $PSScriptRoot "check_db.py") -EvidencePath $dbEvidence -Command "python scripts/check_db.py"
Invoke-Probe -CheckId "W01-API" -ScriptPath (Join-Path $PSScriptRoot "check_api.py") -EvidencePath $apiEvidence -Command "python scripts/check_api.py --mode fake" -Arguments @("--mode", "fake")
Invoke-Probe -CheckId "W01-FS01" -ScriptPath (Join-Path $PSScriptRoot "check_commerce.py") -EvidencePath $fs01Evidence -Command "python scripts/check_commerce.py"
Invoke-Probe -CheckId "W01-T05-FAULTS" -ScriptPath (Join-Path $PSScriptRoot "check_faults.py") -EvidencePath $faultEvidence -Command "python scripts/check_faults.py"

if (-not $env:QUERYSHIELD_BOOTSTRAP_DATABASE_URL) {
    "blocked reason=QUERYSHIELD_BOOTSTRAP_DATABASE_URL_missing" | Set-Content -LiteralPath $fs02Evidence -Encoding UTF8
    Add-CheckResult -CheckId "W01-FS02" -Status "blocked" -Command "python scripts/check_fs02.py" -ExitCode 2 -EvidencePath "W01-FS02.txt"
    Stop-Run -OverallStatus "blocked" -Message "check_blocked id=W01-FS02 evidence=$fs02Evidence" -ExitCode 2
}
Invoke-Probe -CheckId "W01-FS02" -ScriptPath (Join-Path $PSScriptRoot "check_fs02.py") -EvidencePath $fs02Evidence -Command "python scripts/check_fs02.py"

Write-Summary -OverallStatus "pass"
exit 0
