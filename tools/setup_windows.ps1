# One-time Windows setup for Team SPARK, PowerShell version. From the repo root run:
#     powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
# It checks the tools, creates .venv, installs requirements, adds the Hadoop helper
# files PySpark needs on Windows (winutils.exe, hadoop.dll) and sets user
# environment variables (HADOOP_HOME, JAVA_HOME, PYSPARK_PYTHON). Safe to run again.
$ErrorActionPreference = "Continue"
$script:Failed = $false
function Ok($m)  { Write-Host "OK    $m" -ForegroundColor Green }
function Bad($m) { Write-Host "FAIL  $m" -ForegroundColor Red; $script:Failed = $true }
function Warn($m){ Write-Host "WARN  $m" -ForegroundColor Yellow }
function Has($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

Write-Host "== 1. Tools"
if (Has git) { Ok (git --version) } else { Bad "git missing: winget install --id Git.Git -e" }
if (Has gh)  { Ok ((gh --version | Select-Object -First 1)) } else { Bad "gh missing: winget install --id GitHub.cli -e" }
$py = (& py -3.11 --version) 2>&1
if ($LASTEXITCODE -eq 0) { Ok $py } else { Bad "Python 3.11 missing: py install 3.11   (or winget install --id Python.Python.3.11 -e)" }
# Find a Java 17 JDK even if another Java version comes first on PATH.
$javaHome = $null
$cands = @()
if ($env:JAVA_HOME) { $cands += $env:JAVA_HOME }
foreach ($root in "C:\Program Files\Eclipse Adoptium", "C:\Program Files\Java", "C:\Program Files\Microsoft") {
  if (Test-Path $root) { $cands += (Get-ChildItem $root -Directory -Filter "jdk-17*" | ForEach-Object FullName) }
}
if (Has java) { $cands += (Split-Path (Split-Path (Get-Command java).Source -Parent) -Parent) }
foreach ($c in $cands) {
  $jx = Join-Path $c "bin\java.exe"
  if (Test-Path $jx) {
    $jv = (& $jx -version 2>&1 | Select-Object -First 1).ToString()
    if ($jv -match '"17\.') { $javaHome = $c; break }
  }
}
if ($javaHome) { Ok "Java 17 at $javaHome" } else { Bad "Java 17 not found: winget install --id EclipseAdoptium.Temurin.17.JDK -e" }
if (Has docker) { Ok (docker --version) } else { Warn "docker missing (needed for Kafka only): install Docker Desktop" }
if ($script:Failed) { Write-Host "Install what is missing, close and reopen PowerShell, then run this again."; exit 1 }

Write-Host "== 2. Line endings"
git config core.autocrlf input; Ok "core.autocrlf = input (keeps files Linux-style for the team)"

Write-Host "== 3. Python venv"
if (-not (Test-Path .venv)) { & py -3.11 -m venv .venv }
$vpy = Join-Path (Resolve-Path .venv) "Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip
& $vpy -m pip install -q -r requirements.txt
if ($LASTEXITCODE -eq 0) { Ok "requirements installed in .venv" } else { Bad "pip install failed (office proxy? see the hub's help tab)" }

Write-Host "== 4. Hadoop helper files for PySpark on Windows"
$hh = Join-Path $env:USERPROFILE "hadoop"
New-Item -ItemType Directory -Force -Path (Join-Path $hh "bin") | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
foreach ($f in "winutils.exe", "hadoop.dll") {
  $dst = Join-Path $hh "bin\$f"
  if ((Test-Path $dst) -and ((Get-Item $dst).Length -gt 0)) { Ok "$f already present" }
  else {
    try { Invoke-WebRequest -UseBasicParsing -Uri "$base/$f" -OutFile $dst; Ok "downloaded $f" }
    catch { Bad "could not download $f from $base ($($_.Exception.Message))" }
  }
}

Write-Host "== 5. User environment variables"
$vars = @{ HADOOP_HOME = $hh; JAVA_HOME = $javaHome; PYSPARK_PYTHON = "python"; PYSPARK_DRIVER_PYTHON = "python"; PYTHONIOENCODING = "utf-8" }
foreach ($k in $vars.Keys) {
  [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")
  Set-Item -Path "Env:$k" -Value $vars[$k]
}
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$hbin = Join-Path $hh "bin"
if (-not ($userPath -split ";" | Where-Object { $_ -eq $hbin })) {
  [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $hbin), "User")
}
$env:Path = "$hbin;$env:Path"
Ok "HADOOP_HOME=$hh  JAVA_HOME=$javaHome  PYSPARK_PYTHON=python"

Write-Host "== 6. Contract check"
& $vpy tools\fake_data.py | Out-Null
& $vpy tools\check_contracts.py | Select-Object -Last 1

if ($script:Failed) { Write-Host "Setup finished with problems above." -ForegroundColor Red }
else { Write-Host "Setup PASS. Close and reopen PowerShell before running Spark." -ForegroundColor Green }
