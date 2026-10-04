[CmdletBinding()]
param(
    [ValidateSet('fake', 'real')]
    [string]$Mode = 'fake'
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$sourceRoot = Join-Path $projectRoot 'src'

if (-not (Test-Path -LiteralPath $python)) {
    throw "项目虚拟环境不存在：$python"
}

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "源码目录不存在：$sourceRoot"
}

$env:QUERYSHIELD_PROVIDER_MODE = $Mode

& $python -m compileall -q $sourceRoot

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$bootstrapScript = Join-Path $PSScriptRoot 'bootstrap_db.py'

if (-not (Test-Path -LiteralPath $bootstrapScript)) {
    throw "数据库初始化脚本不存在：$bootstrapScript"
}

if ([string]::IsNullOrWhiteSpace($env:QUERYSHIELD_BOOTSTRAP_DATABASE_URL)) {
    throw "QUERYSHIELD_BOOTSTRAP_DATABASE_URL is not configured"
}

& $python $bootstrapScript

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($env:QUERYSHIELD_DATABASE_URL) {
    $databaseState = 'SET'
}
else {
    $databaseState = 'MISSING'
}

Write-Output "bootstrap_ok mode=$Mode database_url=$databaseState"
