from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Content-address source/config versions before each subsequent operation."""
import hashlib
import shutil
from pathlib import Path

def archive():
    root = project_root()
    for p in sorted(list(project_path('scripts', root).glob('*.py')) + list(project_path('configs', root).glob('*.json'))):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        q = project_path('provenance/code_versions', root) / h / p.name
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            if hashlib.sha256(q.read_bytes()).hexdigest() != h:
                raise RuntimeError('Archived code version changed: ' + str(q))
        else:
            shutil.copy2(p, q)
if __name__ == '__main__':
    archive()
