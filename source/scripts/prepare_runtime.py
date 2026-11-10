"""Create a same-platform LAN host bundle; never include research or saved data."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args,**kw): return subprocess.run([str(a) for a in args],cwd=ROOT,check=True,**kw)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=ROOT/'build/runtime')
 p.add_argument('--skip-build',action='store_true');p.add_argument('--offline',action='store_true')
 p.add_argument('--installed-env',type=Path);p.add_argument('--python-source',type=Path)
 args=p.parse_args();stage=args.output.resolve()
 if stage.exists():p.error('Choose a new output directory')
 if not args.skip_build:
  run('cargo','build','--locked','--release','-p','rmuc-sim');run('bun','run','build')
 binary='rmuc-sim.exe' if os.name=='nt' else 'rmuc-sim'
 if not (ROOT/'target/release'/binary).is_file() or not (ROOT/'build/web/apps/client/web/index.html').is_file():p.error('Build engine and web first')
 python_source=args.python_source
 if python_source is None:
  command=['uv','python','find','--managed-python','3.12']+(['--offline'] if args.offline else [])
  exe=Path(subprocess.check_output(command,cwd=ROOT,text=True).strip()).resolve()
  python_source=exe.parent if os.name=='nt' else exe.parent.parent
 python_source=python_source.resolve()
 if not python_source.is_dir() or stage==python_source or stage.is_relative_to(python_source):p.error('Invalid Python source')
 stage.mkdir(parents=True)
 shutil.copytree(python_source,stage/'python',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 python=stage/'python'/('python.exe' if os.name=='nt' else 'bin/python3')
 with (stage/'requirements.txt').open('w') as f:
  run('uv','export','--locked','--package','rmuc2027-server','--no-dev','--no-default-groups','--no-emit-workspace','--format','requirements-txt',stdout=f)
 if args.installed_env:
  original=args.installed_env.resolve()/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
  inspect="import json,sys,sysconfig,platform;print(json.dumps([sysconfig.get_paths()['purelib'],list(sys.version_info[:2]),sys.platform,platform.machine()]))"
  source=json.loads(subprocess.check_output([str(original),'-c',inspect],text=True))
  target=json.loads(subprocess.check_output([str(python),'-c',inspect],text=True))
  if source[1:]!=target[1:]:raise RuntimeError('Python version/platform mismatch')
  shutil.copytree(source[0],target[0],dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','__editable__*','rmuc*-*.dist-info'))
  # A host bundle contains only packages in its exported runtime lock.
  import importlib.metadata
  allowed=set()
  for line in (stage/'requirements.txt').read_text().splitlines():
   if line and not line.startswith((' ','#','-')) and '==' in line:allowed.add(line.split('==')[0].replace('_','-').lower())
  for dist in importlib.metadata.distributions(path=[target[0]]):
   if dist.metadata['Name'].replace('_','-').lower() not in allowed:
    for entry in dist.files or []:
     item=(Path(target[0])/entry).resolve()
     if item.is_file() and item.is_relative_to(stage):item.unlink()
 install=['uv','pip','install','--python',python,'--break-system-packages','--require-hashes','-r',stage/'requirements.txt']
 if args.offline:install.append('--offline')
 run(*install)
 app=stage/'app'
 for name in ('packages/core','apps/server','configs','build/web'):
  shutil.copytree(ROOT/name,app/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
 # Research configs are irrelevant to a fixed-bot LAN host.
 for name in ('rl','migrations'):shutil.rmtree(app/'configs'/name,ignore_errors=True)
 (app/'target/release').mkdir(parents=True);shutil.copy2(ROOT/'target/release'/binary,app/'target/release'/binary)
 shutil.copy2(ROOT/'LICENSE',app/'LICENSE')
 from licenses import collect
 triple=next(line.removeprefix('host: ') for line in subprocess.check_output(['rustc','-vV'],text=True).splitlines() if line.startswith('host: '))
 collect(ROOT,app/'licenses',manifests=('Cargo.toml',),offline=args.offline,target=triple)
 for notice in ('client-assets.json','arena-audio.json','arena-ui-port.json','agent-logos-LICENSE.txt','README.md'):
  shutil.copy2(ROOT/'docs/legal'/notice,app/'licenses'/notice)
 shutil.copy2(ROOT/'apps/client/LICENSE',app/'licenses/CLIENT_LICENSE.txt')
 environment={**os.environ,'RMUC_RESOURCE_ROOT':str(app),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':os.pathsep.join(str(app/name) for name in ('packages/core','apps/server'))}
 subprocess.run([str(python),'-B','-c',"from rmuc2027_core.engine import Match; from rmuc2027_server.control import baseline_helpers; from rmuc2027_server.server import create_app; m=Match(duration=1); m.close(); baseline_helpers(); print('RMUC 2027 runtime preflight OK')"],cwd=app,env=environment,check=True)
 files={str(f.relative_to(stage)).replace(os.sep,'/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(app.rglob('*')) if f.is_file()}
 (stage/'resources.sha256.json').write_text(json.dumps(files,indent=2)+'\n')
 if os.name=='nt':
  (stage/'host.cmd').write_text('@echo off\nset "RMUC_RESOURCE_ROOT=%~dp0app"\nset "PYTHONPATH=%~dp0app\\packages\\core;%~dp0app\\apps\\server"\nif not defined RMUC_USER_DATA set "RMUC_USER_DATA=%LOCALAPPDATA%\\rmuc2027"\ncd /d "%~dp0app"\n"%~dp0python\\python.exe" -B -u -m rmuc2027_server host %*\n')
 else:
  launcher=stage/'host.sh';launcher.write_text('#!/bin/sh\nset -eu\nRMUC_RUNTIME=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexport RMUC_RESOURCE_ROOT="$RMUC_RUNTIME/app"\nexport PYTHONPATH="$RMUC_RUNTIME/app/packages/core:$RMUC_RUNTIME/app/apps/server"\nexport RMUC_USER_DATA="${RMUC_USER_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/rmuc2027}"\ncd "$RMUC_RUNTIME/app"\nexec "$RMUC_RUNTIME/python/bin/python3" -B -u -m rmuc2027_server host "$@"\n');launcher.chmod(0o755)
 print('Runtime ready:',stage)
if __name__=='__main__':main()
