[CmdletBinding()]
param(
    [string]$BailianBaseUrl,
    [string]$EvidenceRoot,
    [ValidateRange(1, 65535)][int]$PostgresPort = 5433,
    [switch]$FakeDryRun,
    [switch]$NonInteractive
)

# B3d demo run: starts a real uvicorn process on the DEMO database
# (queryshield_demo) and drives the demo questions over HTTP, comparing verified
# values with the independently computed expected answers (scripts/demo_run.py).
# Credentials are entered the same way as the HTTP smoke script, live only in
# this process environment, are never written or printed, and are restored when
# the script ends.  The demo setting is given to the server process by demo_run.py,
# not set in your session; an existing QUERYSHIELD_DATABASE_URL (the test database)
# is ignored and restored.  Create the demo database first: docs/b3d-demo-data.md.
# -FakeDryRun checks the wiring with the Fake model at no cost.

$ErrorActionPreference = "Stop"
$previousLocation = Get-Location
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$demoScript = Join-Path $PSScriptRoot "demo_run.py"
$requiredNames = if ($FakeDryRun) {
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
        "QUERYSHIELD_EMBEDDING_DIMENSIONS"
    )
}
# Every name this script may set is saved and restored, including the rerank
# key that the Bailian profile fills alongside the model key.
$touchedNames = @($requiredNames) + @(
    "QUERYSHIELD_DEMO_DATASET",
    "QUERYSHIELD_RERANK_URL",
    "QUERYSHIELD_RERANK_API_KEY",
    "QUERYSHIELD_RERANK_MODEL_NAME"
)
$previousValues = @{}
foreach ($name in $touchedNames) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$exitCode = 0

function Read-B2bSecret {
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
    if ($FakeDryRun -and $BailianBaseUrl) {
        Write-Output 'The Fake dry run does not accept or use a provider URL.'
        $exitCode = 2
        throw [ArgumentException]::new('Provider URL is not applicable to the Fake dry run.')
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

        # Same public service profile as w05-local-real.ps1 (the product B1
        # retriever uses no reranker, so rerank values are not required here).
        $bailianOrigin = $serviceUri.GetLeftPart([UriPartial]::Authority)
        $profileValues = @{
            QUERYSHIELD_MODEL_BASE_URL = "$bailianOrigin/compatible-mode/v1"
            QUERYSHIELD_MODEL_NAME = 'qwen-plus'
            QUERYSHIELD_EMBEDDING_BASE_URL = "$bailianOrigin/compatible-mode/v1"
            QUERYSHIELD_EMBEDDING_MODEL_NAME = 'text-embedding-v4'
            QUERYSHIELD_EMBEDDING_MODEL_REVISION = 'text-embedding-v4'
            QUERYSHIELD_EMBEDDING_DIMENSIONS = '1024'
        }
        foreach ($name in $profileValues.Keys) {
            [Environment]::SetEnvironmentVariable($name, $profileValues[$name], 'Process')
        }

        $sharedKey = [Environment]::GetEnvironmentVariable('QUERYSHIELD_MODEL_API_KEY', 'Process')
        if ([string]::IsNullOrWhiteSpace($sharedKey) -and -not $NonInteractive) {
            $sharedKey = Read-B2bSecret 'Paste the complete Bailian API Key (hidden, not a URL)'
        }
        if (-not [string]::IsNullOrWhiteSpace($sharedKey)) {
            $sharedKey = $sharedKey.Trim()
            if ($sharedKey -notmatch '^sk-[^\s*]+$') {
                Write-Output 'API Key format is invalid; copy the complete Key, not a URL, name or masked asterisks.'
                $exitCode = 2
                throw [ArgumentException]::new('Invalid API Key format.')
            }
            foreach ($name in @('QUERYSHIELD_MODEL_API_KEY', 'QUERYSHIELD_EMBEDDING_API_KEY')) {
                [Environment]::SetEnvironmentVariable($name, $sharedKey, 'Process')
            }
        }
        $sharedKey = $null
        Write-Output 'Bailian profile selected: qwen-plus, text-embedding-v4 (1024). Credentials are process-only.'
    }

    # The demo run always uses the demo database; an ambient test-database URL is never reused.
    $env:QUERYSHIELD_DATABASE_URL = $null
    $env:QUERYSHIELD_DEMO_DATASET = $null
    if (-not [string]::IsNullOrWhiteSpace($env:QUERYSHIELD_DEMO_DATABASE_URL)) {
        $env:QUERYSHIELD_DATABASE_URL = $env:QUERYSHIELD_DEMO_DATABASE_URL
    }
    elseif (-not $NonInteractive) {
        $databasePassword = Read-B2bSecret 'Enter the existing local queryshield_ro password (hidden)'
        if (-not [string]::IsNullOrEmpty($databasePassword)) {
            $escapedPassword = [Uri]::EscapeDataString($databasePassword)
            $env:QUERYSHIELD_DATABASE_URL = "postgresql://queryshield_ro:${escapedPassword}@127.0.0.1:${PostgresPort}/queryshield_demo"
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
    foreach ($name in $(if (-not $NonInteractive -and -not $BailianBaseUrl) { $emptyNames })) {
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
        $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss-fff", [Globalization.CultureInfo]::InvariantCulture)
        # -EvidenceRoot is the base directory; the run directory below reuses the name (PowerShell variables ignore case) once it has been read.
        $evidenceBase = if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) { Join-Path $projectRoot "evidence" } else { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EvidenceRoot) }
        $mode = if ($FakeDryRun) { 'fake' } else { 'real' }
        $evidenceRoot = Join-Path $evidenceBase "B3d-demo-$mode-$stamp"
        New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null
        Set-Location -LiteralPath $projectRoot
        Write-Output "Running the B3d demo questions ($mode) against a local uvicorn process on the demo database; output is question ids, verdicts and known gaps only. The raw file (b3d-demo-raw.json) holds answers and customer names: do not share it."
        & $pythonPath $demoScript --mode $mode --evidence-dir $evidenceRoot
        $demoExitCode = $LASTEXITCODE
        Write-Output "Evidence root: $evidenceRoot"
        $exitCode = if ($demoExitCode -eq 0) { 0 } elseif ($demoExitCode -eq 2) { 2 } else { 1 }
    }
}
catch {
    # Avoid printing exception messages that could contain a local endpoint or secret.
    Write-Output ("B3d demo run stopped after a sanitized failure (" + $_.Exception.GetType().Name + "). Credentials will be restored.")
    if ($exitCode -ne 2) { $exitCode = 1 }
}
finally {
    $sharedKey = $null
    $databasePassword = $null
    $escapedPassword = $null
    foreach ($name in $touchedNames) {
        [Environment]::SetEnvironmentVariable($name, $previousValues[$name], "Process")
    }
    Set-Location -LiteralPath $previousLocation.Path
}

exit $exitCode
