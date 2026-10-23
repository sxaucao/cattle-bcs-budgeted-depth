"""Repository-relative paths and discovery of optional system dependencies."""
from pathlib import Path
import json,os,shutil

def project_root():
    return Path(__file__).resolve().parents[1]

def project_path(value,root=None):
    base=Path(root) if root is not None else project_root()
    p=Path(value)
    if p.is_absolute():return p
    text=str(value).replace('\\','/')
    head,sep,tail=text.partition('/')
    layout=json.loads((base/'layout.json').read_text(encoding='utf-8'))
    if head in layout:return base/layout[head]/tail if sep else base/layout[head]
    return base/p

def relative_record(value):
    p=project_path(value).resolve()
    return Path(os.path.relpath(p,project_root())).as_posix()

def get_executable(name):
    hit=shutil.which(name)
    if hit:return hit
    local=project_root()/'02_code/tools'/name
    if local.is_file():return str(local)
    raise FileNotFoundError(f'{name} must be installed and discoverable on PATH; no machine-specific location is assumed')

def find_font(bold=False):
    name='timesbd.ttf' if bold else 'times.ttf'
    env=os.environ.get('BCS_TNR_BOLD' if bold else 'BCS_TNR_REGULAR')
    candidates=[project_root()/'02_code/resources/fonts'/name]
    if env:candidates.insert(0,project_path(env))
    if os.environ.get('WINDIR'):candidates.append(Path(os.environ['WINDIR'])/'Fonts'/name)
    for p in candidates:
        if p.is_file():return str(p)
    raise FileNotFoundError('Install Times New Roman or set BCS_TNR_REGULAR and BCS_TNR_BOLD; proprietary fonts are not redistributed')
