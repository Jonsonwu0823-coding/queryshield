[CmdletBinding()]
param(
    [string]$BailianBaseUrl,
    [string]$EvidenceDir,
    [ValidateRange(30, 900)][int]$StartupTimeoutSeconds = 240,
    [switch]$FakeDryRun,
    [ValidateSet('json', 'native')][string]$ModelProtocol = 'json',
    [switch]$NonInteractive
)

# The demo questions and the walkthrough against the Docker Compose stack with the REAL model.
#
# Needs Docker Desktop. Enter the Bailian workspace address and key the same way as demo-local.ps1:
# they live only in this process environment (docker compose passes them to the app container),
# are never written to a file or printed, and are restored when the script ends.
# -FakeDryRun runs the same steps with the Fake model at no cost (no address or key needed).
#
# Steps: create .env if it is missing (random credentials, never printed) -> docker compose up -d --build
# -> wait for the app to be healthy -> docker compose exec ... demo_run.py --base-url ... and
# demo_walkthrough.py -> copy the two summaries to the evidence folder -> docker compose down
# (volumes are kept; "docker compose down -v" resets everything).
#
# The summaries hold ids, status codes, terminal states and numbers only. The raw demo file
# (demo-raw.json, answers and customer names) stays inside the container and is not copied.

$ErrorActionPreference = "Stop"
$previousLocation = Get-Location
$projectRoot = Split-Path -Parent $PSScriptRoot
$touchedNames = @(
    "QUERYSHIELD_PROVIDER_MODE",
    "QUERYSHIELD_MODEL_PROTOCOL",
    "QUERYSHIELD_MODEL_BASE_URL",
    "QUERYSHIELD_MODEL_API_KEY",
    "QUERYSHIELD_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_BASE_URL",
    "QUERYSHIELD_EMBEDDING_API_KEY",
    "QUERYSHIELD_EMBEDDING_MODEL_NAME",
    "QUERYSHIELD_EMBEDDING_MODEL_REVISION",
    "QUERYSHIELD_EMBEDDING_DIMENSIONS",
    "QUERYSHIELD_METADATA_TOOLS"
)
$previousValues = @{}
foreach ($name in $touchedNames) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$exitCode = 0
$stackStarted = $false

function Read-HiddenSecret {
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

function Invoke-Docker {
    # Returns ONLY the integer exit code. The command's output (docker compose up --build writes the build log
    # to standard output) goes to the screen: output left in the pipeline would be returned together with the
    # exit code, and "(Invoke-Docker ...) -ne 0" would then filter that array instead of comparing a number.
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker @Arguments | Out-Host
        return [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-DockerText {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& docker @Arguments 2>$null)
        return @{ ExitCode = $LASTEXITCODE; Text = (($output | ForEach-Object { [string]$_ }) -join "`n") }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    if ($FakeDryRun -and $BailianBaseUrl) {
        Write-Output 'The Fake dry run does not accept or use a provider URL.'
        $exitCode = 2
        throw [ArgumentException]::new('Provider URL is not applicable to the Fake dry run.')
    }
    if (-not $FakeDryRun -and -not $BailianBaseUrl) {
        Write-Output 'Give -BailianBaseUrl (the Beijing workspace OpenAI compatible address from the Bailian API Key page), or use -FakeDryRun.'
        $exitCode = 2
        throw [ArgumentException]::new('Provider URL is missing.')
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Output 'docker was not found. Start Docker Desktop and open a new PowerShell window.'
        $exitCode = 2
        throw [ArgumentException]::new('docker is missing.')
    }

    $mode = if ($FakeDryRun) { 'fake' } else { 'real' }
    [Environment]::SetEnvironmentVariable("QUERYSHIELD_PROVIDER_MODE", $mode, "Process")
    [Environment]::SetEnvironmentVariable("QUERYSHIELD_MODEL_PROTOCOL", $ModelProtocol, "Process")
    # The demo run is judged against the local metadata tools unless you set the MCP setting yourself.
    if ($mode -eq 'real') {
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
            $sharedKey = Read-HiddenSecret 'Paste the complete Bailian API Key (hidden, not a URL)'
        }
        if ([string]::IsNullOrWhiteSpace($sharedKey)) {
            Write-Output 'The API Key is not available; nothing was started.'
            $exitCode = 2
            throw [ArgumentException]::new('API key missing.')
        }
        $sharedKey = $sharedKey.Trim()
        if ($sharedKey -notmatch '^sk-[^\s*]+$') {
            Write-Output 'API Key format is invalid; copy the complete Key, not a URL, name or masked asterisks.'
            $exitCode = 2
            throw [ArgumentException]::new('Invalid API Key format.')
        }
        foreach ($name in @('QUERYSHIELD_MODEL_API_KEY', 'QUERYSHIELD_EMBEDDING_API_KEY')) {
            [Environment]::SetEnvironmentVariable($name, $sharedKey, 'Process')
        }
        $sharedKey = $null
        Write-Output 'Bailian profile selected: qwen-plus, text-embedding-v4 (1024). Credentials are process-only.'
    }

    Set-Location -LiteralPath $projectRoot

    # .env holds random database passwords and tokens (never printed). Model keys are not stored in it.
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env") -PathType Leaf)) {
        $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
        $pythonCommand = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { "python" }
        & $pythonCommand (Join-Path $PSScriptRoot "new_env.py")
        if ($LASTEXITCODE -ne 0) {
            Write-Output 'Could not create .env. See docs/operations.md for the docker run alternative.'
            $exitCode = 2
            throw [InvalidOperationException]::new('.env was not created.')
        }
    }

    $stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss-fff", [Globalization.CultureInfo]::InvariantCulture)
    if ([string]::IsNullOrWhiteSpace($EvidenceDir)) {
        $EvidenceDir = Join-Path (Join-Path $projectRoot "evidence") "compose-$mode-$stamp"
    }
    New-Item -ItemType Directory -Path $EvidenceDir -Force | Out-Null

    Write-Output "Building the image and starting the stack ($mode model). The first build downloads the dependencies."
    $stackStarted = $true
    if ((Invoke-Docker @("compose", "up", "-d", "--build")) -ne 0) {
        Write-Output 'docker compose up failed. Is Docker Desktop running? The port is QUERYSHIELD_HTTP_PORT (default 8000).'
        $exitCode = 2
        throw [InvalidOperationException]::new('compose up failed.')
    }

    $healthy = $false
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $idResult = Get-DockerText @("compose", "ps", "-q", "app")
        if ($idResult.ExitCode -eq 0 -and $idResult.Text.Trim()) {
            $health = Get-DockerText @("inspect", "-f", "{{.State.Health.Status}}", $idResult.Text.Trim())
            if ($health.Text.Trim() -eq "healthy") {
                $healthy = $true
                break
            }
        }
        Start-Sleep -Seconds 3
    }
    if (-not $healthy) {
        Write-Output 'The app did not become healthy in time. Check: docker compose logs app'
        $exitCode = 2
        throw [InvalidOperationException]::new('app not healthy.')
    }

    Write-Output "Running the demo questions ($mode) inside the app container; output is question ids, verdicts and known gaps only."
    $demoExit = Invoke-Docker @("compose", "exec", "-T", "app", "python", "scripts/demo_run.py", "--mode", $mode, "--base-url", "http://127.0.0.1:8000", "--model-protocol", $ModelProtocol, "--evidence-dir", "/tmp/demo-run")
    Write-Output "Running the walkthrough ($mode)."
    $walkExit = Invoke-Docker @("compose", "exec", "-T", "app", "python", "scripts/demo_walkthrough.py", "--mode", $mode, "--evidence-dir", "/tmp/walkthrough")

    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $copies = @(
        @{ From = "/tmp/demo-run/demo-summary.json"; To = "demo-summary.json" },
        @{ From = "/tmp/walkthrough/walkthrough-summary.json"; To = "walkthrough-summary.json" }
    )
    foreach ($copy in $copies) {
        $result = Get-DockerText @("compose", "exec", "-T", "app", "cat", $copy.From)
        if ($result.ExitCode -eq 0 -and $result.Text) {
            [System.IO.File]::WriteAllText((Join-Path $EvidenceDir $copy.To), $result.Text + "`n", $utf8)
        }
        else {
            Write-Output ("Could not read " + $copy.To + " from the container.")
        }
    }
    Write-Output "Evidence root: $EvidenceDir"
    if ($demoExit -eq 2 -or $walkExit -eq 2) { $exitCode = 2 }
    elseif ($demoExit -ne 0 -or $walkExit -ne 0) { $exitCode = 1 }
    else { $exitCode = 0 }
}
catch {
    # Avoid printing exception messages that could contain a local endpoint or secret.
    Write-Output ("Compose demo stopped after a sanitized failure (" + $_.Exception.GetType().Name + "). Credentials will be restored.")
    if ($exitCode -eq 0) { $exitCode = 1 }
}
finally {
    $sharedKey = $null
    if ($stackStarted) {
        Write-Output "Stopping the stack (docker compose down; the volumes are kept)."
        [void](Invoke-Docker @("compose", "down"))
    }
    foreach ($name in $touchedNames) {
        [Environment]::SetEnvironmentVariable($name, $previousValues[$name], "Process")
    }
    Set-Location -LiteralPath $previousLocation.Path
}

exit $exitCode
