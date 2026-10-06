[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceDir,
    [string]$PythonPath,
    [ValidateRange(1, 65535)][int]$PostgresPort = 5433,
    [switch]$NonInteractive
)

# The one check entry shared by the local machine and CI.
#
# Runs, in order: the full test suite; DB-SMOKE, BASE, PROPOSAL (fake), AGENT, STATE and EVAL (fake)
# through scripts/check.ps1; then the Fake smokes (HTTP, MCP, demo questions).  The
# results go into one summary (check-all-summary.json); the exit code is 0 only when every
# step passed or is recorded as not applicable with its reason.
#
# Checks that cannot run everywhere are listed once, below, each with its reason.  The list
# plus the checks actually invoked must equal what check.ps1 registers (tests/test_check_all.py
# reads this file and check.ps1; this script re-checks it against every check.ps1 summary).
#
# Database settings: QUERYSHIELD_DATABASE_URL (read-only role, queryshield_test) and
# QUERYSHIELD_BOOTSTRAP_DATABASE_URL (owner role, same database).  On Windows, when they are
# not set, the two passwords are asked for hidden and the URLs point at 127.0.0.1:<PostgresPort>.
# Identity tokens that are not set are generated in this process.  Nothing is printed or
# written; every environment change is undone when the script ends.

$ErrorActionPreference = "Stop"
$scriptStarted = [DateTime]::UtcNow
$projectRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $projectRoot
$isWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
$checkScript = Join-Path $PSScriptRoot "check.ps1"
$hostExecutable = (Get-Process -Id $PID).Path

# Checks that check.ps1 registers but this script does not run unconditionally.
#   excluded    never run here
#   conditional run when the condition holds, otherwise recorded as not_applicable
$script:ExcludedChecks = @(
    @{ Id = "EVAL-R01"; Suite = "EVAL"; Kind = "excluded"; Reason = "needs the sealed holdout set, which exists only on the local machine and is not in the repository; the local full mode (eval-local-real.ps1) covers it" },
    @{ Id = "EVAL-R06"; Suite = "EVAL"; Kind = "excluded"; Reason = "needs the sealed holdout commitments, which exist only on the local machine; the local full mode (eval-local-real.ps1) covers it" },
    @{ Id = "STATE-X01"; Suite = "STATE"; Kind = "conditional"; Reason = "reads the upstream asset register and the acceptance tags the register lists; not applicable when the register or a listed tag is missing (for example in an exported repository without this history); a git that cannot run is a failure, not a missing tag" },
    @{ Id = "EVAL-X01"; Suite = "EVAL"; Kind = "conditional"; Reason = "reads the upstream asset register and the acceptance tags the register lists; not applicable when the register or a listed tag is missing (for example in an exported repository without this history); a git that cannot run is a failure, not a missing tag" }
)
$script:RegisterRelativePath = "control/evidence/upstream/accepted-assets.json"
# The CI workflow inside the project directory means the project directory is itself the repository root
# (the standalone layout, as in tests/repo_layout.py); the register belongs to the development repository only.
$script:StandaloneWorkflowRelativePath = ".github/workflows/queryshield-ci.yml"
# Suites run through check.ps1, in order.  STATE and EVAL get an explicit -CheckIds list
# (registered ids minus the exclusions); the others run in full.
$script:CheckSuites = @("DB-SMOKE", "BASE", "PROPOSAL", "AGENT", "STATE", "EVAL")

$script:steps = @()
$script:consistency = [ordered]@{}
$script:savedEnvironment = @{}

function Test-AbsolutePath {
    param([string]$Path)
    if ($isWindowsHost) {
        return ($Path -match '^[A-Za-z]:[\\/]') -or ($Path -match '^\\\\')
    }
    return $Path -match '^/'
}

function Set-EnvironmentValue {
    # $null removes the variable (setting an empty string would leave it defined).
    param([string]$Name, $Value)
    if ($null -eq $Value) {
        Remove-Item -LiteralPath ("Env:" + $Name) -ErrorAction SilentlyContinue
    }
    else {
        Set-Item -LiteralPath ("Env:" + $Name) -Value ([string]$Value)
    }
}

function Set-ProcessEnvironment {
    param([string]$Name, $Value)
    if (-not $script:savedEnvironment.ContainsKey($Name)) {
        $script:savedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
    }
    Set-EnvironmentValue -Name $Name -Value $Value
}

function Restore-ProcessEnvironment {
    foreach ($name in @($script:savedEnvironment.Keys)) {
        Set-EnvironmentValue -Name $name -Value $script:savedEnvironment[$name]
    }
    $script:savedEnvironment = @{}
}

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

function New-RandomToken {
    $bytes = New-Object byte[] 24
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return (($bytes | ForEach-Object { $_.ToString("x2") }) -join "")
}

function Convert-DatabaseName {
    param([string]$Url, [string]$Name)
    return ($Url -replace '/[^/?]+(\?.*)?$', ('/' + $Name + '$1'))
}

function Add-Step {
    param([string]$Name, [string]$Status, [int]$ExitCode, [string]$Reason, [string]$Evidence, [hashtable]$Extra)
    $record = [ordered]@{
        name = $Name
        status = $Status
        exit_code = $ExitCode
        reason = $Reason
        evidence = $Evidence
    }
    if ($Extra) {
        foreach ($key in $Extra.Keys) {
            $record[$key] = $Extra[$key]
        }
    }
    $script:steps += $record
    Write-Output ("check_all step={0} status={1} exit_code={2}" -f $Name, $Status, $ExitCode)
}

function Write-StepStart {
    # One line before each step, so a long step does not look like a hang.
    param([string]$Name, [string]$Note)
    $line = "check_all running step=" + $Name
    if ($Note) { $line += " (" + $Note + ")" }
    Write-Output $line
}

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments, [string]$OutputFile)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location -LiteralPath $projectRoot
    try {
        $output = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $ErrorActionPreference = $previousPreference
    }
    if ($output.Count -eq 0) {
        $output = @("no_output")
    }
    @($output | ForEach-Object { [string]$_ }) | Set-Content -LiteralPath $OutputFile -Encoding UTF8
    return $exitCode
}

function Get-RegisteredCheckIds {
    param([string]$Suite)
    $text = [System.IO.File]::ReadAllText($checkScript)
    $pattern = '"' + [regex]::Escape($Suite) + '"\s*\{\s*@\(([^)]*)\)\s*\}'
    $match = [regex]::Match($text, $pattern)
    if (-not $match.Success) {
        return @()
    }
    return @([regex]::Matches($match.Groups[1].Value, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
}

function Test-X01Applicable {
    # Returns @{ Applicable; Failure; Reason }.
    #   the project directory is itself the repository root (the CI workflow is inside it): not applicable
    #   no register in this checkout, or a tag the register lists is missing: not applicable
    #   git cannot run (not installed, dubious ownership, not a repository, ...): a FAILURE; the X01 checks
    #   still run so that their own evidence is there, and the failure is recorded as its own step.
    # The tag names come from the register (the same ones the X01 checks read), never from this script.
    if (Test-Path -LiteralPath (Join-Path $projectRoot $script:StandaloneWorkflowRelativePath) -PathType Leaf) {
        return @{ Applicable = $false; Failure = $false; Reason = "this directory is the repository root (the CI workflow is inside it); the upstream asset register is not part of it, and a register in a parent directory is not looked for" }
    }
    $register = Join-Path $repoRoot $script:RegisterRelativePath
    if (-not (Test-Path -LiteralPath $register -PathType Leaf)) {
        return @{ Applicable = $false; Failure = $false; Reason = "the upstream asset register is not in this checkout" }
    }
    try {
        $document = Read-JsonFile -Path $register
        $tagNames = @($document.tags.PSObject.Properties | ForEach-Object { $_.Name })
    }
    catch {
        return @{ Applicable = $true; Failure = $true; Reason = "the upstream asset register cannot be read as JSON" }
    }
    if ($tagNames.Count -eq 0) {
        return @{ Applicable = $true; Failure = $true; Reason = "the upstream asset register lists no acceptance tags" }
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return @{ Applicable = $true; Failure = $true; Reason = "git is not available to look up the acceptance tags" }
    }
    # Same isolation as the X01 checks: a caller's repository-routing variables are ignored, and only this
    # repository root is trusted for the read-only lookup (the checkout can belong to another account).
    $routing = @("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM")
    $savedRouting = @{}
    foreach ($name in $routing) {
        $savedRouting[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        Remove-Item -LiteralPath ("Env:" + $name) -ErrorAction SilentlyContinue
    }
    $safeDirectory = $repoRoot.Replace("\", "/")
    $missing = @()
    $gitFailure = $null
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        foreach ($tag in $tagNames) {
            [void](& git -c "safe.directory=$safeDirectory" -C $repoRoot rev-parse --verify --quiet "refs/tags/$tag^{commit}" 2>$null)
            $code = $LASTEXITCODE
            if ($code -eq 1) {
                $missing += $tag
            }
            elseif ($code -ne 0) {
                $gitFailure = "git could not read the repository (exit code $code)"
                break
            }
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
        foreach ($name in $routing) {
            if ($null -ne $savedRouting[$name]) {
                Set-Item -LiteralPath ("Env:" + $name) -Value $savedRouting[$name]
            }
        }
    }
    if ($gitFailure) {
        return @{ Applicable = $true; Failure = $true; Reason = $gitFailure }
    }
    if ($missing.Count -gt 0) {
        return @{ Applicable = $false; Failure = $false; Reason = ("acceptance tag(s) missing: " + ($missing -join ", ")) }
    }
    return @{ Applicable = $true; Failure = $false; Reason = "" }
}

function Read-JsonFile {
    param([string]$Path)
    return [System.IO.File]::ReadAllText($Path) | ConvertFrom-Json
}

function Invoke-CheckSuite {
    param([string]$Suite, [bool]$X01Applicable, [string]$X01Reason)

    $dir = Join-Path $EvidenceDir $Suite
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-StepStart -Name $Suite -Note "check.ps1 -Suite $Suite, usually under a minute"
    $arguments = @()
    if ($isWindowsHost) {
        $arguments += @("-ExecutionPolicy", "Bypass")
    }
    $arguments += @("-NoProfile", "-File", $checkScript, "-Suite", $Suite, "-Mode", "fake", "-Database", "postgres", "-EvidenceDir", $dir)
    if ($PythonPath) {
        $arguments += @("-PythonPath", $PythonPath)
    }

    $excludedHere = @($script:ExcludedChecks | Where-Object { $_.Suite -eq $Suite })
    $notRun = @()
    foreach ($item in $excludedHere) {
        if ($item.Kind -eq "excluded") {
            $notRun += $item
        }
        elseif (-not $X01Applicable) {
            $notRun += $item
        }
    }
    if ($notRun.Count -gt 0) {
        $registered = Get-RegisteredCheckIds -Suite $Suite
        if ($registered.Count -eq 0) {
            Add-Step -Name $Suite -Status "fail" -ExitCode 1 -Reason "check.ps1 does not register an id list for this suite" -Evidence $Suite
            return
        }
        $skipIds = @($notRun | ForEach-Object { $_.Id })
        $selected = @($registered | Where-Object { $skipIds -notcontains $_ })
        $arguments += @("-CheckIds", ($selected -join ","))
    }

    $outputFile = Join-Path $dir "check-output.txt"
    $exitCode = Invoke-Native -FilePath $hostExecutable -Arguments $arguments -OutputFile $outputFile

    $summaryPath = Join-Path $dir "summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        Add-Step -Name $Suite -Status "fail" -ExitCode $exitCode -Reason "check.ps1 wrote no summary.json" -Evidence $Suite
        return
    }
    $summary = Read-JsonFile -Path $summaryPath
    $required = @($summary.all_required_check_ids)
    $executed = @($summary.checks | ForEach-Object { $_.check_id })
    $passing = @($summary.checks | Where-Object { $_.status -in @("pass", "not_applicable") } | ForEach-Object { $_.check_id })
    $failing = @($summary.checks | Where-Object { $_.status -notin @("pass", "not_applicable") } | ForEach-Object { "$($_.check_id):$($_.status)" })

    # Invoked plus excluded must equal what check.ps1 registers for this suite, both ways.
    $excludedIds = @($excludedHere | ForEach-Object { $_.Id })
    $notRunIds = @($notRun | ForEach-Object { $_.Id })
    $expectedRun = @($required | Where-Object { $notRunIds -notcontains $_ })
    $missingChecks = @($expectedRun | Where-Object { $passing -notcontains $_ })
    $unexpectedChecks = @($executed | Where-Object { $notRunIds -contains $_ })
    $unknownExclusions = @($excludedIds | Where-Object { $required -notcontains $_ })
    $script:consistency[$Suite] = [ordered]@{
        registered = $required
        run = $expectedRun
        not_run = @($notRun | ForEach-Object { $_.Id })
        missing = $missingChecks
        unexpected = $unexpectedChecks
        exclusion_not_registered = $unknownExclusions
    }

    $reasons = @()
    if ($exitCode -ne 0) { $reasons += "check.ps1 exit code $exitCode" }
    if ($failing.Count -gt 0) { $reasons += ("not passing: " + ($failing -join ", ")) }
    if ($missingChecks.Count -gt 0) { $reasons += ("registered checks not run: " + ($missingChecks -join ", ")) }
    if ($unexpectedChecks.Count -gt 0) { $reasons += ("excluded checks were run: " + ($unexpectedChecks -join ", ")) }
    if ($unknownExclusions.Count -gt 0) { $reasons += ("exclusion not registered by check.ps1: " + ($unknownExclusions -join ", ")) }
    $status = if ($reasons.Count -eq 0) { "pass" } else { "fail" }
    Add-Step -Name $Suite -Status $status -ExitCode $exitCode -Reason ($reasons -join "; ") -Evidence $Suite -Extra @{ checks_passed = $passing.Count; checks_not_run = @($notRun | ForEach-Object { $_.Id }) }

    foreach ($item in $notRun) {
        if ($item.Kind -eq "conditional") {
            $reason = $X01Reason
        }
        else {
            $reason = $item.Reason
        }
        Add-Step -Name $item.Id -Status "not_applicable" -ExitCode 0 -Reason $reason -Evidence $Suite
    }
}

function Test-X02Reasons {
    # EVAL-X02 must return 2 for the reasons it tests, not for an evidence path rule.
    $dir = Join-Path $EvidenceDir "EVAL"
    $expectations = @(
        @{ Prefix = "unknown_check_id-"; Text = "unknown_CheckIds=EVAL-UNKNOWN" },
        @{ Prefix = "missing_mode-"; Text = "EVAL_Mode_must_be_explicit" }
    )
    $problems = @()
    $seen = [ordered]@{}
    foreach ($item in $expectations) {
        $verdict = "ok"
        $folders = @(Get-ChildItem -LiteralPath $dir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like ($item.Prefix + "*") })
        $blocked = $null
        if ($folders.Count -gt 0) {
            $blocked = Join-Path $folders[0].FullName "check-blocked.txt"
        }
        if ($folders.Count -eq 0) {
            $verdict = "folder_missing"
        }
        elseif (-not (Test-Path -LiteralPath $blocked -PathType Leaf)) {
            $verdict = "no_check_blocked"
        }
        else {
            $text = [System.IO.File]::ReadAllText($blocked)
            if ($text -match "evidence_dir_must_be_absolute") {
                $verdict = "blocked_by_evidence_path_rule"
            }
            elseif ($text -notmatch [regex]::Escape($item.Text)) {
                $verdict = "reason_missing"
            }
        }
        $seen[$item.Prefix] = $verdict
        if ($verdict -ne "ok") {
            $problems += ($item.Prefix + " " + $verdict)
        }
    }
    $verdictAll = if ($problems.Count -eq 0) { "pass" } else { "fail" }
    $script:consistency["x02_reason_check"] = [ordered]@{ status = $verdictAll; details = $seen }
    Add-Step -Name "EVAL-X02-reasons" -Status $verdictAll -ExitCode $(if ($problems.Count -eq 0) { 0 } else { 1 }) -Reason ($problems -join "; ") -Evidence "EVAL"
}

function Write-CheckAllSummary {
    param([string]$OverallStatus, [hashtable]$PythonInfo)
    $summary = [ordered]@{
        suite = "check-all"
        mode = "fake"
        platform = $(if ($isWindowsHost) { "windows" } else { "non-windows" })
        python = $PythonInfo
        started_at = $scriptStarted.ToString("o")
        ended_at = [DateTime]::UtcNow.ToString("o")
        excluded_checks = @($script:ExcludedChecks | ForEach-Object { [ordered]@{ id = $_.Id; kind = $_.Kind; reason = $_.Reason } })
        consistency = $script:consistency
        steps = @($script:steps)
        overall_status = $OverallStatus
    }
    $path = Join-Path $EvidenceDir "check-all-summary.json"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
    Write-Output "summary_path=$path"
}

# ----------------------------------------------------------------------------------------

if (-not (Test-AbsolutePath -Path $EvidenceDir)) {
    Write-Output "check_all_blocked reason=evidence_dir_must_be_absolute"
    exit 2
}

if ($PythonPath) {
    $python = $PythonPath
    $pythonSource = "explicit"
}
elseif ($isWindowsHost) {
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $pythonSource = "default"
}
else {
    $python = Join-Path (Join-Path (Join-Path $projectRoot ".venv") "bin") "python"
    $pythonSource = "default"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Output "check_all_blocked reason=python_missing"
    exit 2
}

New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$exitCode = 0

try {
    # The whole run, tests and smokes included, works without the MCP setting.
    Set-ProcessEnvironment -Name "QUERYSHIELD_METADATA_TOOLS" -Value $null

    $missingNames = @()
    foreach ($name in @("QUERYSHIELD_DATABASE_URL", "QUERYSHIELD_BOOTSTRAP_DATABASE_URL")) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
            $missingNames += $name
        }
    }
    if ($missingNames.Count -gt 0 -and $isWindowsHost -and -not $NonInteractive) {
        $roPassword = Read-HiddenSecret "Enter the local queryshield_ro password (hidden)"
        $adminPassword = Read-HiddenSecret "Enter the local queryshield owner password (hidden)"
        if ($missingNames -contains "QUERYSHIELD_DATABASE_URL" -and -not [string]::IsNullOrEmpty($roPassword)) {
            Set-ProcessEnvironment -Name "QUERYSHIELD_DATABASE_URL" -Value ("postgresql://queryshield_ro:{0}@127.0.0.1:{1}/queryshield_test" -f [Uri]::EscapeDataString($roPassword), $PostgresPort)
        }
        if ($missingNames -contains "QUERYSHIELD_BOOTSTRAP_DATABASE_URL" -and -not [string]::IsNullOrEmpty($adminPassword)) {
            Set-ProcessEnvironment -Name "QUERYSHIELD_BOOTSTRAP_DATABASE_URL" -Value ("postgresql://queryshield:{0}@127.0.0.1:{1}/queryshield_test" -f [Uri]::EscapeDataString($adminPassword), $PostgresPort)
        }
        $roPassword = $null
        $adminPassword = $null
    }
    $stillMissing = @()
    foreach ($name in @("QUERYSHIELD_DATABASE_URL", "QUERYSHIELD_BOOTSTRAP_DATABASE_URL")) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
            $stillMissing += $name
        }
    }
    if ($stillMissing.Count -gt 0) {
        Write-Output ("check_all_blocked reason=configuration_missing names=" + ($stillMissing -join ","))
        $exitCode = 2
        throw [ArgumentException]::new("configuration missing")
    }

    $tokenNames = @("QUERYSHIELD_TOKEN_A_REQUESTER", "QUERYSHIELD_TOKEN_A_APPROVER", "QUERYSHIELD_TOKEN_B_REQUESTER", "QUERYSHIELD_TOKEN_B_APPROVER")
    $anyTokenMissing = @($tokenNames | Where-Object { [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_, "Process")) }).Count -gt 0
    if ($anyTokenMissing) {
        foreach ($name in $tokenNames) {
            Set-ProcessEnvironment -Name $name -Value (New-RandomToken)
        }
    }

    $testUrl = $env:QUERYSHIELD_DATABASE_URL
    $demoUrl = Convert-DatabaseName -Url $testUrl -Name "queryshield_demo"

    # Interpreter record (check.ps1 evidence does not carry it).
    $pythonInfoOutput = @(& $python -c "import sys; print(sys.version.split()[0]); print(sys.platform)" 2>$null)
    $relative = $python
    $rootPrefix = $projectRoot.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    if ($python.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $python.Substring($rootPrefix.Length).Replace("\", "/")
    }
    else {
        $relative = [System.IO.Path]::GetFileName($python)
    }
    $pythonInfo = [ordered]@{
        version = $(if ($pythonInfoOutput.Count -ge 1) { [string]$pythonInfoOutput[0] } else { "unknown" })
        platform = $(if ($pythonInfoOutput.Count -ge 2) { [string]$pythonInfoOutput[1] } else { "unknown" })
        path = $relative
        source = $pythonSource
    }

    # 1. Full test suite.
    $testDir = Join-Path $EvidenceDir "tests"
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    $testOutput = Join-Path $testDir "pytest-output.txt"
    # pytest's default temporary directory (<temp>/pytest-of-<user>) may belong to another account, which
    # fails every test that uses tmp_path. Use a directory created for this run, under the system temp
    # directory (not the evidence directory), and delete it afterwards.
    $testBaseTemp = Join-Path ([System.IO.Path]::GetTempPath()) ("qs-check-all-" + [Guid]::NewGuid().ToString("N"))
    Write-StepStart -Name "tests" -Note "the full test suite: about 2 to 5 minutes, about 4 on Windows; no output until it ends"
    try {
        $testExit = Invoke-Native -FilePath $python -Arguments @("-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "--basetemp", $testBaseTemp) -OutputFile $testOutput
    }
    finally {
        Remove-Item -LiteralPath $testBaseTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
    $testText = [System.IO.File]::ReadAllText($testOutput)
    $passedCount = 0
    $skippedCount = 0
    $failedCount = 0
    $passedMatch = [regex]::Match($testText, '(\d+) passed')
    if ($passedMatch.Success) { $passedCount = [int]$passedMatch.Groups[1].Value }
    $skippedMatch = [regex]::Match($testText, '(\d+) skipped')
    if ($skippedMatch.Success) { $skippedCount = [int]$skippedMatch.Groups[1].Value }
    $failedMatch = [regex]::Match($testText, '(\d+) failed')
    if ($failedMatch.Success) { $failedCount = [int]$failedMatch.Groups[1].Value }
    # Tests that need the demo database must run here, not skip.
    $demoSkips = 0
    foreach ($line in ($testText -split "`n")) {
        if ($line -match "^SKIPPED \[(\d+)\].*demo database") { $demoSkips += [int]$Matches[1] }
    }
    $testReasons = @()
    if ($testExit -ne 0) { $testReasons += "pytest exit code $testExit" }
    if ($demoSkips -gt 0) { $testReasons += "$demoSkips test(s) that need the demo database were skipped" }
    Add-Step -Name "tests" -Status $(if ($testReasons.Count -eq 0) { "pass" } else { "fail" }) -ExitCode $testExit -Reason ($testReasons -join "; ") -Evidence "tests" -Extra @{ passed = $passedCount; skipped = $skippedCount; failed = $failedCount; skipped_demo_database = $demoSkips }
    Write-Output ("check_all tests passed={0} skipped={1} failed={2} skipped_demo_database={3}" -f $passedCount, $skippedCount, $failedCount, $demoSkips)

    # 2. check.ps1 suites.
    Write-StepStart -Name "x01-lookup" -Note "looking up the acceptance tags the register lists"
    $x01 = Test-X01Applicable
    if ($x01.Failure) {
        Add-Step -Name "X01-applicability" -Status "fail" -ExitCode 1 -Reason $x01.Reason -Evidence ""
    }
    foreach ($suite in $script:CheckSuites) {
        Invoke-CheckSuite -Suite $suite -X01Applicable $x01.Applicable -X01Reason $x01.Reason
    }
    $x02Step = @($script:steps | Where-Object { $_.name -eq "EVAL" })
    if ($x02Step.Count -gt 0) {
        $evalSummary = Join-Path (Join-Path $EvidenceDir "EVAL") "summary.json"
        if (Test-Path -LiteralPath $evalSummary -PathType Leaf) {
            $ran = @((Read-JsonFile -Path $evalSummary).checks | Where-Object { $_.check_id -eq "EVAL-X02" -and $_.status -eq "pass" }).Count -gt 0
            if ($ran) { Test-X02Reasons }
        }
    }

    # 3. Fake smokes.
    $smokes = @(
        @{ Name = "smoke-http"; Dir = "smoke-http"; Script = "http_smoke.py"; Args = @("--mode", "fake"); Demo = $false },
        @{ Name = "smoke-http-native"; Dir = "smoke-http-native"; Script = "http_smoke.py"; Args = @("--mode", "fake", "--model-protocol", "native"); Demo = $false },
        @{ Name = "smoke-mcp"; Dir = "smoke-mcp"; Script = "mcp_smoke.py"; Args = @("--mode", "fake", "--part", "all"); Demo = $false },
        @{ Name = "demo-run"; Dir = "demo-run"; Script = "demo_run.py"; Args = @("--mode", "fake"); Demo = $true }
    )
    foreach ($smoke in $smokes) {
        $dir = Join-Path $EvidenceDir $smoke.Dir
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-StepStart -Name $smoke.Name -Note "a few seconds to about a minute"
        $arguments = @((Join-Path $PSScriptRoot $smoke.Script)) + $smoke.Args + @("--evidence-dir", $dir)
        if ($smoke.Demo) {
            Set-ProcessEnvironment -Name "QUERYSHIELD_DATABASE_URL" -Value $demoUrl
        }
        $smokeExit = Invoke-Native -FilePath $python -Arguments $arguments -OutputFile (Join-Path $dir "output.txt")
        if ($smoke.Demo) {
            Set-ProcessEnvironment -Name "QUERYSHIELD_DATABASE_URL" -Value $testUrl
        }
        Add-Step -Name $smoke.Name -Status $(if ($smokeExit -eq 0) { "pass" } else { "fail" }) -ExitCode $smokeExit -Reason $(if ($smokeExit -eq 0) { "" } else { "exit code $smokeExit" }) -Evidence $smoke.Dir
    }

    $failedSteps = @($script:steps | Where-Object { $_.status -notin @("pass", "not_applicable") })
    $overall = if ($failedSteps.Count -eq 0) { "pass" } else { "fail" }
    $exitCode = if ($failedSteps.Count -eq 0) { 0 } else { 1 }
    Write-CheckAllSummary -OverallStatus $overall -PythonInfo $pythonInfo
    Write-Output ("check_all overall={0}" -f $overall)
    foreach ($step in $failedSteps) {
        Write-Output ("check_all failed step={0} reason={1}" -f $step.name, $step.reason)
    }
}
catch [System.ArgumentException] {
    # Blocked before any check ran; the message was printed above.
    if ($exitCode -eq 0) { $exitCode = 2 }
}
catch {
    # Never print the message: it could hold a connection string.
    Write-Output ("check_all internal_error type=" + $_.Exception.GetType().Name)
    Add-Step -Name "check-all-internal-error" -Status "fail" -ExitCode 1 -Reason $_.Exception.GetType().Name -Evidence ""
    Write-CheckAllSummary -OverallStatus "fail" -PythonInfo $(if ($pythonInfo) { $pythonInfo } else { @{} })
    $exitCode = 1
}
finally {
    Restore-ProcessEnvironment
}
exit $exitCode
