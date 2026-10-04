[CmdletBinding()]
param(
    [ValidateSet('fake', 'real')]
    [string]$Mode = 'fake',
    [ValidateRange(1, 65535)]
    [int]$Port = 18080,
    [string]$GatewayBaseUrl
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$env:QUERYSHIELD_PROVIDER_MODE = $Mode
if ($PSBoundParameters.ContainsKey('GatewayBaseUrl')) {
    $env:QUERYSHIELD_GATEWAY_BASE_URL = $GatewayBaseUrl
}

& $python -m uvicorn queryshield.api.main:app `
    --app-dir (Join-Path $projectRoot 'src') `
    --host 127.0.0.1 `
    --port $Port
exit $LASTEXITCODE
