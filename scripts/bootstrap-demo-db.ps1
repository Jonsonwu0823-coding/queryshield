[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$PostgresPort = 5433,
    [string]$AdminUser = 'queryshield',
    [string]$DatabaseName = 'queryshield_demo',
    [switch]$NonInteractive
)

# B3d: create the demo tables and rows in an existing *_demo database
# (scripts/bootstrap_demo_db.py).  Create the database first, for example:
#   docker exec queryshield-postgres-w01 psql -U queryshield -d postgres -c "CREATE DATABASE queryshield_demo"
# The administrator password is entered hidden (or taken from
# QUERYSHIELD_DEMO_ADMIN_PASSWORD with -NonInteractive), lives only in this
# process environment, is never printed, and the previous environment is restored.
# The read-only role queryshield_ro must already exist (README); this script
# neither creates roles nor sets passwords.

$ErrorActionPreference = "Stop"
$previousLocation = Get-Location
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$bootstrapScript = Join-Path $PSScriptRoot "bootstrap_demo_db.py"
$touchedNames = @("QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL")
$previousValues = @{}
foreach ($name in $touchedNames) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$exitCode = 0

try {
    if (-not $DatabaseName.EndsWith('_demo') -or $DatabaseName.EndsWith('_test')) {
        Write-Output 'The database name must end with _demo.'
        $exitCode = 2
        throw [ArgumentException]::new('Invalid database name.')
    }
    $password = [Environment]::GetEnvironmentVariable("QUERYSHIELD_DEMO_ADMIN_PASSWORD", "Process")
    if ([string]::IsNullOrEmpty($password) -and -not $NonInteractive) {
        $secure = Read-Host -Prompt "Enter the local $AdminUser password (hidden)" -AsSecureString
        $pointer = [IntPtr]::Zero
        try {
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($secure)
            $password = [Runtime.InteropServices.Marshal]::PtrToStringUni($pointer)
        }
        finally {
            if ($pointer -ne [IntPtr]::Zero) {
                [Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($pointer)
            }
            $secure.Dispose()
        }
    }
    if ([string]::IsNullOrEmpty($password)) {
        Write-Output 'The administrator password is not available; nothing was run.'
        $exitCode = 2
    }
    else {
        $escaped = [Uri]::EscapeDataString($password)
        $env:QUERYSHIELD_DEMO_BOOTSTRAP_DATABASE_URL = "postgresql://${AdminUser}:${escaped}@127.0.0.1:${PostgresPort}/${DatabaseName}"
        $password = $null
        $escaped = $null
        Set-Location -LiteralPath $projectRoot
        & $pythonPath $bootstrapScript
        $exitCode = if ($LASTEXITCODE -eq 0) { 0 } elseif ($LASTEXITCODE -eq 2) { 2 } else { 1 }
    }
}
catch {
    # Avoid printing exception messages that could contain a local endpoint or secret.
    Write-Output ("Demo database bootstrap stopped after a sanitized failure (" + $_.Exception.GetType().Name + "). Credentials will be restored.")
    if ($exitCode -ne 2) { $exitCode = 1 }
}
finally {
    $password = $null
    $escaped = $null
    foreach ($name in $touchedNames) {
        [Environment]::SetEnvironmentVariable($name, $previousValues[$name], "Process")
    }
    Set-Location -LiteralPath $previousLocation.Path
}

exit $exitCode
