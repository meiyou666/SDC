$ErrorActionPreference = "Stop"
$sdcArguments = @($args)
$sdcRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $sdcRoot

$sdcDockerCommand = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$sdcDocker = if ($sdcDockerCommand) { $sdcDockerCommand.Source } else { $null }
if (-not $sdcDocker) { throw "Docker CLI was not found. Complete the Windows setup in README, then reopen PowerShell." }
$env:Path = (Split-Path $sdcDocker -Parent) + ";" + $env:Path

# Docker's registry client does not read Windows system proxy settings itself.
# These changes apply only to this launcher and its child processes.
if (-not $env:HTTPS_PROXY) {
    $sdcRegistry = [uri]"https://auth.docker.io"
    $sdcProxyProvider = [Net.WebRequest]::DefaultWebProxy
    if ($sdcProxyProvider) {
        $sdcProxy = $sdcProxyProvider.GetProxy($sdcRegistry)
        if ($sdcProxy -and $sdcProxy -ne $sdcRegistry) {
            $env:HTTPS_PROXY = $sdcProxy.AbsoluteUri
            if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $sdcProxy.AbsoluteUri }
        }
    }
}

$ErrorActionPreference = "Continue"
& $sdcDocker info *> $null
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("Docker Engine is unavailable. Check the selected Docker context and engine connection.")
    exit 1
}
& $sdcDocker compose version *> $null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$sdcComposeArgs = @("compose", "run", "--build", "--rm")
if ($sdcArguments.Count -gt 0) { $sdcComposeArgs += "-T" }
$sdcComposeArgs += "dev"
$sdcComposeArgs += $sdcArguments
& $sdcDocker @sdcComposeArgs
exit $LASTEXITCODE
