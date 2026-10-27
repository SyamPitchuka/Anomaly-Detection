#!/usr/bin/env bash
# One-time Windows setup for Team SPARK. Run in Git Bash from the repo root:
#     bash tools/setup_windows.sh
# It checks the tools, creates .venv, installs requirements, adds the Hadoop helper
# files PySpark needs on Windows (winutils.exe, hadoop.dll), and writes the
# environment lines to ~/.bashrc. Safe to run again.
set -u
ok()   { printf 'OK    %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; FAILED=1; }
FAILED=0

echo "== 1. Tools"
command -v git >/dev/null && ok "git $(git --version | cut -d' ' -f3)" || bad "git missing: winget install --id Git.Git -e"
command -v gh  >/dev/null && ok "gh $(gh --version | head -1 | cut -d' ' -f3)" || bad "gh missing: winget install --id GitHub.cli -e"
if py -3.11 --version >/dev/null 2>&1; then ok "$(py -3.11 --version)"; else bad "Python 3.11 missing: winget install --id Python.Python.3.11 -e"; fi
JV=$(java -version 2>&1 | head -1)
case "$JV" in *'"17.'*) ok "Java 17 ($JV)";; *) bad "Java 17 missing or wrong ($JV): winget install --id EclipseAdoptium.Temurin.17.JDK -e";; esac
command -v docker >/dev/null && ok "docker $(docker --version | cut -d' ' -f3 | tr -d ,)" || printf 'WARN  docker missing (needed for Kafka only): install Docker Desktop\n'
[ "$FAILED" = 1 ] && { echo "Install what is missing, close and reopen Git Bash, then run this again."; exit 1; }

echo "== 2. Line endings"
git config core.autocrlf input && ok "core.autocrlf = input (keeps files Linux-style for the team)"

echo "== 3. Python venv"
[ -d .venv ] || py -3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/Scripts/activate && ok "venv active: $(python --version)"
python -m pip install -q --upgrade pip && pip install -q -r requirements.txt && ok "requirements installed" || bad "pip install failed (office proxy? see the hub's help tab)"

echo "== 4. Hadoop helper files for PySpark on Windows"
HH="$HOME/hadoop"
mkdir -p "$HH/bin"
BASE="https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
for f in winutils.exe hadoop.dll; do
  if [ -s "$HH/bin/$f" ]; then ok "$f already present"
  else curl -fsSL -o "$HH/bin/$f" "$BASE/$f" && ok "downloaded $f" || bad "could not download $f from $BASE"
  fi
done

echo "== 5. Environment (written to ~/.bashrc once)"
# Java and Hadoop read these as Windows paths (C:\...), so convert them with cygpath.
JH=$(cygpath -w "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")")
HW=$(cygpath -w "$HH")
MARK="# >>> spark-anomaly >>>"
if ! grep -q "$MARK" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF
$MARK
export HADOOP_HOME="$HW"
export PATH="\$HOME/hadoop/bin:\$PATH"
export JAVA_HOME="$JH"
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python
export PYTHONIOENCODING=utf-8
# <<< spark-anomaly <<<
EOF
  ok "added HADOOP_HOME, JAVA_HOME, PYSPARK_PYTHON to ~/.bashrc"
else ok "~/.bashrc already set"; fi
export HADOOP_HOME="$HW" PATH="$HOME/hadoop/bin:$PATH" JAVA_HOME="$JH" PYSPARK_PYTHON=python PYSPARK_DRIVER_PYTHON=python PYTHONIOENCODING=utf-8

echo "== 6. Contract check"
python tools/fake_data.py >/dev/null && python tools/check_contracts.py | tail -1

[ "$FAILED" = 1 ] && echo "Setup finished with problems above." || echo "Setup PASS. Open a new Git Bash window before running Spark."
