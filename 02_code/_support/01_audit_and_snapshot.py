from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Create a byte-verified snapshot and auditable, append-only data accounting.

Read-only against the source project. A fresh timestamped audit directory is used
on every execution; the snapshot never overwrites a differing existing file.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = project_root()
CONFIG = project_path('configs/data_protocol_v1.json', ROOT)
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def csv_out(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def csv_in(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', str(s))]

def main():
    from importlib import import_module
    import_module('00_archive_sources').archive()
    started = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    src = Path(cfg['source_dataset'])
    legacy = Path(cfg['legacy_project'])
    dest = project_path('data/raw_snapshot/dataset', ROOT)
    audit_id = datetime.now().strftime('%Y%m%dT%H%M%S')
    out = project_path('provenance', ROOT) / ('audit_' + audit_id)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG, out / CONFIG.name)
    files = sorted((p for p in src.rglob('*') if p.is_file()), key=lambda p: natural(p.relative_to(src)))
    total = sum((p.stat().st_size for p in files))
    if shutil.disk_usage(ROOT).free < total + 2 * 2 ** 30:
        raise RuntimeError('Not enough free space for snapshot and safety margin')
    sources = project_path('provenance/sources', ROOT)
    source_rows = []
    selected = [legacy / n for n in ['README.md', 'README_EDITOR_REVISION.md', '01_build_new_rgb_depth_index_and_folds.py', '02_make_longitudinal_protocol_splits.py', '01_build_new_rgb_depth_index_and_folds/01_new_rgb_depth_frame_index.csv', '01_build_new_rgb_depth_index_and_folds/01_new_rgb_depth_bag_inventory.csv', '02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv', '02_make_longitudinal_protocol_splits/02_identity_audit.csv', '09_editor_revision_results/split_manifest.csv']]
    selected.append(Path(cfg['previous_article_pdf']))
    for p in selected:
        if not p.exists():
            continue
        rel = p.relative_to(legacy) if p.is_relative_to(legacy) else Path('previous_article') / p.name
        q = sources / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        h = sha256(p)
        if q.exists() and sha256(q) != h:
            raise RuntimeError(f'Provenance source changed: {q}')
        if not q.exists():
            shutil.copy2(p, q)
        source_rows.append({'original_path': str(p), 'snapshot_path': q.relative_to(ROOT).as_posix(), 'sha256': h, 'bytes': p.stat().st_size})
    csv_out(out / 'source_documents.csv', source_rows, ['original_path', 'snapshot_path', 'sha256', 'bytes'])
    inventory = []
    pairfiles = defaultdict(lambda: defaultdict(list))
    issues = []
    for i, p in enumerate(files):
        rel = p.relative_to(src)
        q = dest / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        before = p.stat()
        h = sha256(p)
        if q.exists():
            if sha256(q) != h:
                raise RuntimeError(f'Refusing to overwrite differing snapshot file: {q}')
        else:
            shutil.copy2(p, q)
        copied_hash = sha256(q)
        after = p.stat()
        if copied_hash != h or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Copy/source stability check failed: {p}')
        parts = rel.parts
        info = {'source_relpath': rel.as_posix(), 'snapshot_relpath': q.relative_to(ROOT).as_posix(), 'bytes': before.st_size, 'mtime_utc': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(), 'sha256_source': h, 'sha256_snapshot': copied_hash, 'copy_verified': True, 'bcs': '', 'episode': '', 'animal_group': '', 'frame_number': '', 'modality': 'unrecognised'}
        m = re.fullmatch('f_(\\d+)_(rgb|depth)', p.stem, flags=re.IGNORECASE)
        if len(parts) == 3 and parts[0] in ['2', '3', '4', '5', '6'] and m:
            animal = re.match('^Cow_(\\d+)(?:_|$)', parts[1], flags=re.IGNORECASE)
            if animal:
                mod = m[2].lower()
                if mod == 'rgb' and p.suffix.lower() in IMG_EXT or (mod == 'depth' and p.suffix.lower() == '.npy'):
                    info.update(bcs=int(parts[0]), episode=f'BCS{parts[0]}__{parts[1]}', animal_group=f'Cow_{int(animal[1])}', frame_number=int(m[1]), modality=mod)
                    pairfiles[info['episode'], int(m[1])][mod].append(info)
        if info['modality'] == 'unrecognised':
            issues.append({'unit': rel.as_posix(), 'type': 'unrecognised_file', 'detail': 'Not silently included'})
        inventory.append(info)
        if (i + 1) % 1000 == 0:
            print(f'Copied and SHA-256 verified {i + 1}/{len(files)} files', flush=True)
    csv_out(out / 'files_manifest.csv', inventory, list(inventory[0]))
    extra_dest = {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} - {p.relative_to(src).as_posix() for p in files}
    for path in extra_dest:
        issues.append({'unit': path, 'type': 'extra_snapshot_file', 'detail': 'Not present in source'})
    legacy_rows = csv_in(sources / '02_make_longitudinal_protocol_splits/02A_animalwise_3fold.csv')
    old = {}
    for r in legacy_rows:
        key = (r['bag_id'], int(r['frame_key']))
        if key in old:
            issues.append({'unit': str(key), 'type': 'legacy_duplicate_key', 'detail': ''})
        old[key] = r
    pairs = []
    decoded_groups = defaultdict(list)
    for key, mods in sorted(pairfiles.items(), key=lambda kv: natural(str(kv[0]))):
        ep, frame = key
        if len(mods['rgb']) != 1 or len(mods['depth']) != 1:
            issues.append({'unit': str(key), 'type': 'ambiguous_or_missing_pair', 'detail': f"RGB={len(mods['rgb'])},depth={len(mods['depth'])}"})
            continue
        rgb, dep = (mods['rgb'][0], mods['depth'][0])
        row = {'pair_id': f'{ep}__f{frame}', 'episode_id': ep, 'animal_group': rgb['animal_group'], 'bcs': rgb['bcs'], 'class_index': rgb['bcs'] - 2, 'frame_number': frame, 'rgb_relpath': rgb['snapshot_relpath'], 'depth_relpath': dep['snapshot_relpath'], 'rgb_sha256': rgb['sha256_source'], 'depth_sha256': dep['sha256_source'], 'outer_fold': '', 'rgb_valid': False, 'depth_valid': False, 'shape_match': False, 'rgb_shape': '', 'rgb_mode': '', 'rgb_decoded_sha256': '', 'depth_shape': '', 'depth_dtype': '', 'depth_valid_fraction': '', 'depth_zero_fraction': '', 'depth_nonfinite_count': '', 'depth_p01': '', 'depth_p99': '', 'depth_min': '', 'depth_max': '', 'depth_constant': '', 'depth_units': 'unverified', 'status': '', 'reason': ''}
        reasons = []
        try:
            with Image.open(ROOT / row['rgb_relpath']) as image:
                image.load()
                row['rgb_mode'] = image.mode
                a = np.asarray(image.convert('RGB'))
            row['rgb_shape'] = 'x'.join(map(str, a.shape))
            row['rgb_decoded_sha256'] = hashlib.sha256(row['rgb_shape'].encode() + a.tobytes()).hexdigest()
            row['rgb_valid'] = True
            decoded_groups[row['rgb_decoded_sha256']].append(row)
        except Exception as exc:
            reasons.append(f'rgb_decode_error:{type(exc).__name__}')
        try:
            d = np.load(ROOT / row['depth_relpath'], allow_pickle=False)
            row['depth_shape'] = 'x'.join(map(str, d.shape))
            row['depth_dtype'] = str(d.dtype)
            if d.ndim != 2 or not np.issubdtype(d.dtype, np.number) or np.iscomplexobj(d):
                raise ValueError('Depth must be a real numeric HxW array')
            valid = np.isfinite(d) & (d > 0)
            row['depth_valid_fraction'] = float(valid.mean())
            row['depth_zero_fraction'] = float((d == 0).mean())
            row['depth_nonfinite_count'] = int((~np.isfinite(d)).sum())
            if valid.any():
                v = d[valid].astype('float64')
                row['depth_min'], row['depth_max'] = (float(v.min()), float(v.max()))
                row['depth_p01'], row['depth_p99'] = map(float, np.percentile(v, [1, 99]))
                row['depth_constant'] = row['depth_p99'] <= row['depth_p01']
                row['depth_valid'] = not row['depth_constant']
                if row['depth_constant']:
                    reasons.append('constant_valid_depth')
            else:
                reasons.append('no_positive_finite_depth')
            if row['rgb_valid']:
                row['shape_match'] = a.shape[:2] == d.shape
                if not row['shape_match']:
                    reasons.append('rgb_depth_spatial_shape_mismatch')
        except Exception as exc:
            reasons.append(f'depth_load_error:{type(exc).__name__}')
        prior = old.get(key)
        if prior is None:
            reasons.append('not_in_legacy_index')
        elif prior['animal_id'] != row['animal_group'] or int(prior['bcs_raw']) != row['bcs'] or prior['rgb_file'] != Path(row['rgb_relpath']).name or (prior['depth_file'] != Path(row['depth_relpath']).name):
            reasons.append('legacy_metadata_disagreement')
        else:
            row['outer_fold'] = int(prior['fold_id'])
        row['status'] = 'eligible' if not reasons else 'needs_resolution'
        row['reason'] = ';'.join(reasons)
        if reasons:
            issues.append({'unit': row['pair_id'], 'type': 'pair_integrity', 'detail': row['reason']})
        pairs.append(row)
    missing_old = set(old) - set(pairfiles)
    for k in missing_old:
        issues.append({'unit': str(k), 'type': 'missing_legacy_pair', 'detail': ''})
    csv_out(out / 'pairs_audit.csv', pairs, list(pairs[0]))
    dup = []
    for h, group in decoded_groups.items():
        if len(group) > 1:
            crossgroup = len({r['animal_group'] for r in group}) > 1
            for r in group:
                dup.append({'decoded_sha256': h, 'pair_id': r['pair_id'], 'animal_group': r['animal_group'], 'bcs': r['bcs'], 'outer_fold': r['outer_fold'], 'duplicate_group_size': len(group), 'cross_animal_group': crossgroup, 'label_conflict': len({g['bcs'] for g in group}) > 1})
            if crossgroup or len({g['bcs'] for g in group}) > 1:
                issues.append({'unit': h, 'type': 'cross_group_or_label_duplicate', 'detail': json.dumps([r['pair_id'] for r in group])})
    csv_out(out / 'exact_rgb_duplicates.csv', dup, ['decoded_sha256', 'pair_id', 'animal_group', 'bcs', 'outer_fold', 'duplicate_group_size', 'cross_animal_group', 'label_conflict'])
    episodes = []
    by_ep = defaultdict(list)
    for r in pairs:
        by_ep[r['episode_id']].append(r)
    for ep, group in by_ep.items():
        for col in ['bcs', 'animal_group', 'outer_fold']:
            if len({r[col] for r in group}) != 1:
                issues.append({'unit': ep, 'type': 'episode_metadata_conflict', 'detail': col})
        episodes.append({'episode_id': ep, 'animal_group': group[0]['animal_group'], 'bcs': group[0]['bcs'], 'outer_fold': group[0]['outer_fold'], 'pair_count': len(group), 'eligible_pairs': sum((r['status'] == 'eligible' for r in group)), 'first_frame_number': min((r['frame_number'] for r in group)), 'last_frame_number': max((r['frame_number'] for r in group)), 'frame_numbers_are_not_timestamps': True})
    csv_out(out / 'episodes_audit.csv', episodes, list(episodes[0]))
    for animal in {r['animal_group'] for r in pairs}:
        if len({r['outer_fold'] for r in pairs if r['animal_group'] == animal}) != 1:
            issues.append({'unit': animal, 'type': 'animal_fold_leakage', 'detail': ''})
    distributions = []
    for c in range(2, 7):
        g = [r for r in pairs if r['bcs'] == c]
        distributions.append({'bcs': c, 'pairs': len(g), 'eligible_pairs': sum((r['status'] == 'eligible' for r in g)), 'episodes': len({r['episode_id'] for r in g}), 'animal_groups': len({r['animal_group'] for r in g})})
    csv_out(out / 'class_counts.csv', distributions, list(distributions[0]))
    csv_out(out / 'issues.csv', issues, ['unit', 'type', 'detail'])
    stages = [{'stage': 'H0', 'operation': 'Historical synchronized frame extraction', 'unit': 'RGB-D pairs', 'input': '', 'output': 5940, 'excluded': '', 'evidence': 'previous_article_sec_2.1.1_Table1', 'status': 'reported_not_file_reconstructed'}, {'stage': 'H1', 'operation': 'Historical manual availability screening', 'unit': 'RGB-D pairs', 'input': 5940, 'output': 3815, 'excluded': 2125, 'evidence': 'previous_article_sec_2.1.1_Table1', 'status': 'reported_no_per_pair_exclusion_manifest_found_in_supplied_project'}, {'stage': 'S0', 'operation': 'Supplied prepared source inventory', 'unit': 'files', 'input': len(files), 'output': len(files), 'excluded': 0, 'evidence': 'files_manifest.csv', 'status': 'measured'}, {'stage': 'S1', 'operation': 'Copy and SHA-256 verification', 'unit': 'files', 'input': len(files), 'output': len(inventory), 'excluded': 0, 'evidence': 'files_manifest.csv', 'status': 'measured'}, {'stage': 'S2', 'operation': 'Unambiguous RGB-depth pairing', 'unit': 'RGB-D pairs', 'input': len(pairfiles), 'output': len(pairs), 'excluded': len(pairfiles) - len(pairs), 'evidence': 'pairs_audit.csv;issues.csv', 'status': 'measured'}, {'stage': 'S3', 'operation': 'Decode, numeric-depth and metadata audit', 'unit': 'RGB-D pairs', 'input': len(pairs), 'output': sum((r['status'] == 'eligible' for r in pairs)), 'excluded': sum((r['status'] != 'eligible' for r in pairs)), 'evidence': 'pairs_audit.csv;issues.csv', 'status': 'measured_flagged_not_deleted'}, {'stage': 'S4', 'operation': 'Episode and animal-group indexing', 'unit': 'episodes / animal groups', 'input': len(pairs), 'output': f"{len(episodes)} / {len({r['animal_group'] for r in pairs})}", 'excluded': 0, 'evidence': 'episodes_audit.csv', 'status': 'measured_grouping_inherited'}]
    csv_out(out / 'stage_ledger.csv', stages, list(stages[0]))
    summary = {'audit_id': audit_id, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'source_dataset': str(src), 'snapshot_dataset': str(dest), 'source_bytes': total, 'source_files': len(files), 'verified_files': len(inventory), 'paired_frames': len(pairs), 'eligible_pairs': sum((r['status'] == 'eligible' for r in pairs)), 'episodes': len(episodes), 'animal_groups': len({r['animal_group'] for r in pairs}), 'issues': len(issues), 'duplicate_rgb_rows': len(dup), 'training_gate': 'PASS' if not issues else 'STOP_REVIEW_ISSUES', 'rgb_shapes': dict(Counter((r['rgb_shape'] for r in pairs))), 'depth_dtypes': dict(Counter((r['depth_dtype'] for r in pairs))), 'python': sys.version, 'platform': platform.platform(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'Pillow']}, 'script_sha256': sha256(Path(__file__)), 'config_sha256': sha256(CONFIG), 'historical_evidence': cfg['historical_counts'], 'historical_exclusion_manifest': 'not located in the supplied legacy project; no reconstructed exclusion reasons assigned', 'scope': 'Exact-content/structural audit; not manual anatomical quality adjudication or near-duplicate/physical-identity verification.'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs = [{'file': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    (out / 'audit_checksums.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    project_path('provenance/latest_audit.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if issues:
        raise SystemExit(2)
