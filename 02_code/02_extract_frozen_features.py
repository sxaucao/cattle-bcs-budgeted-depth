import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Optional raw-input re-extraction for the self-contained main-study release."""
import json
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module
from collections import defaultdict
import numpy as np
import torch
fd = import_module('19_foundation_pooling')
u = fd.u
ROOT = fd.ROOT
ml = import_module('28_multilevel_depth')

def main():
    import_module('00_archive_sources').archive()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = ROOT / project_path('provenance/latest_foundation_experiment.txt', ROOT).read_text().strip()
    prepared = ROOT / project_path('provenance/latest_prepared.txt', ROOT).read_text().strip()
    epids = sorted((r['episode_id'] for r in u.csv_in(prepared / 'episodes.csv')))
    pos = {e: i for i, e in enumerate(epids)}
    run = project_path('runs', ROOT) / ('foundation_refresh_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir()
    net, ac = fd.get_model()
    features = {}
    frame = {}
    records = {}
    for mod in ['RGB', 'Depth']:
        records[mod] = u.csv_in(old / ('source_' + mod + '_tokens') / 'input_manifest.csv')
        rr = []
        for r in records[mod]:
            path = Path(r['path'])
            assert path.exists(), str(path)
            rr.append({'episode_id': r['episode_id'], 'item': r['item'], 'path': relative_record(path), 'depth': mod == 'Depth'})
        folder = run / ('source_' + mod + '_tokens')
        folder.mkdir()
        tok, cl, v, c = fd.extract(net, rr, folder)
        pooled, selector = fd.pool(tok, cl, v, c, rr)
        features['DINO_' + mod + '_global_SVR'] = np.stack([pooled['global'][e] for e in epids])
        patch = np.stack([fd.norm(fd.norm(np.asarray(tok[i], dtype='float32'))[v[i]].mean(0)) for i in range(len(tok))])
        frame[mod] = (fd.norm(cl), patch)
        records[mod] = rr
    features['DINO_RGBD_global_SVR'] = np.concatenate([features['DINO_RGB_global_SVR'], features['DINO_Depth_global_SVR']], 1) / np.sqrt(2)
    np.savez_compressed(run / 'source_episode_features.npz', episode_ids=np.array(epids), **features)
    by = defaultdict(list)
    for i, r in enumerate(records['RGB']):
        by[r['episode_id']].append(i)
    levels = {}
    cost = np.zeros((151, 5), int)
    for col, k in enumerate([1, 3, 6, 12], 1):
        vals = []
        for ep in epids:
            rr = np.array(by[ep])
            order = ml.order_frames(np.zeros((len(rr), 1)), 'temporal_spread')
            ix = rr[order[:min(k, len(rr))]]
            d = np.concatenate([fd.norm(frame['Depth'][0][ix].mean(0)), fd.norm(frame['Depth'][1][ix].mean(0))]) / np.sqrt(2)
            vals.append(np.concatenate([features['DINO_RGB_global_SVR'][pos[ep]], d]) / np.sqrt(2))
            cost[pos[ep], col] = len(ix)
        levels['frames_' + str(k)] = np.asarray(vals, dtype='float64')
    inputs = project_path('runs', ROOT) / ('multilevel_inputs_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    inputs.mkdir()
    np.savez_compressed(inputs / 'temporal_spread_features.npz', episode_ids=np.array(epids), costs=cost, **levels)
    info = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'events': 151, 'selected_frames': 1245, 'encoder_frozen': True, 'weights_sha256': ac['weights_sha256'], 'source_input_manifest': old.relative_to(ROOT).as_posix(), 'script_sha256': u.sha256(Path(__file__))}
    (run / 'COMPLETE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (inputs / 'COMPLETE.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    project_path('provenance/latest_foundation_experiment.txt', ROOT).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    project_path('provenance/latest_multilevel_depth.txt', ROOT).write_text(inputs.relative_to(ROOT).as_posix(), encoding='utf-8')
    print('REFRESHED', run, inputs, flush=True)
if __name__ == '__main__':
    main()
