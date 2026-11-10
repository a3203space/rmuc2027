param([string]$Output = 'build\client-dist', [string]$Runtime = '', [string]$QtPrefix = '')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (Test-Path $Output) { throw 'Choose a new output directory.' }
$ConfigureArgs = @('--preset', 'release')
if ($QtPrefix) { $ConfigureArgs += "-DCMAKE_PREFIX_PATH=$QtPrefix" }
& cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& cmake --build --preset release
if ($LASTEXITCODE -ne 0) { throw 'Client build failed.' }
$Executable = @('build\client\bin\Release\RMUC2027Client.exe','build\client\bin\RMUC2027Client.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Executable) { throw 'Built executable not found.' }
$DeployQt = if ($QtPrefix) { Join-Path $QtPrefix 'bin\windeployqt.exe' } else { 'windeployqt' }
New-Item -ItemType Directory $Output | Out-Null
Copy-Item $Executable $Output
Copy-Item (Join-Path (Split-Path $Executable -Parent) 'web') (Join-Path $Output 'web') -Recurse
& $DeployQt --release --compiler-runtime (Join-Path $Output 'RMUC2027Client.exe')
if ($LASTEXITCODE -ne 0) { throw 'Qt deployment failed.' }
Copy-Item apps\client\LICENSE (Join-Path $Output 'CLIENT_LICENSE.txt')
Copy-Item LICENSE (Join-Path $Output 'LICENSE.txt')
Copy-Item docs\legal\client-assets.json (Join-Path $Output 'CLIENT_ASSETS.json')
Copy-Item docs\legal\arena-audio.json (Join-Path $Output 'arena-audio.json')
Copy-Item docs\legal\README.md (Join-Path $Output 'THIRD_PARTY_NOTICES.md')
Copy-Item docs\lan-play.md (Join-Path $Output '联机说明.md')
if ($Runtime) {
 if (-not (Test-Path (Join-Path $Runtime 'app\apps\server\rmuc2027_server\server.py'))) { throw 'Not a prepared RMUC 2027 runtime.' }
 Copy-Item $Runtime (Join-Path $Output 'runtime') -Recurse
}
Write-Host "Deployment ready: $Output"
