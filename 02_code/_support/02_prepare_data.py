from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Resolve documented structural flags, build a traceable cache and group splits.

No training or model-output-based data selection is performed by this script.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap
from importlib import import_module
audit_helpers = import_module('01_audit_and_snapshot')
sha256, csv_out, csv_in = (audit_helpers.sha256, audit_helpers.csv_out, audit_helpers.csv_in)
ROOT = project_root()

def letterbox(array, shape, interpolation):
    th, tw = shape
    h, w = array.shape[:2]
    ratio = min(th / h, tw / w)
    nh, nw = (max(1, int(round(h * ratio))), max(1, int(round(w * ratio))))
    image = Image.fromarray(array)
    image = image.resize((nw, nh), resample=interpolation)
    result_shape = (th, tw, array.shape[2]) if array.ndim == 3 else (th, tw)
    result = np.zeros(result_shape, dtype=array.dtype)
    top, left = ((th - nh) // 2, (tw - nw) // 2)
    result[top:top + nh, left:left + nw] = np.asarray(image)
    return (result, {'scale': ratio, 'resized_h': nh, 'resized_w': nw, 'pad_top': top, 'pad_left': left})

def choose_inner_groups(episodes, outer, cfg):
    train_candidates = [e for e in episodes if int(e['outer_fold']) != outer]
    groups = sorted({e['animal_group'] for e in train_candidates})
    nval = max(1, int(math.ceil(len(groups) * cfg['fraction_groups'])))
    rng = np.random.default_rng(cfg['seed'] + outer)
    totals = Counter((int(e['bcs']) for e in train_candidates))
    best = None
    for _ in range(cfg['candidate_count']):
        val = set(rng.choice(groups, nval, replace=False).tolist())
        vc = Counter((int(e['bcs']) for e in train_candidates if e['animal_group'] in val))
        score = (sum((totals[c] - vc[c] == 0 for c in range(2, 7))), sum((vc[c] == 0 for c in range(2, 7))), sum(((vc[c] / totals[c] - cfg['fraction_groups']) ** 2 for c in totals)))
        if best is None or score < best[0]:
            best = (score, val)
    return (best, nval)

def main():
    import_module('00_archive_sources').archive()
    cfgpath = project_path('configs/preprocessing_v1.json', ROOT)
    cfg = json.loads(cfgpath.read_text(encoding='utf-8'))
    audit = ROOT / project_path('provenance/latest_audit.txt', ROOT).read_text(encoding='utf-8').strip()
    for r in json.loads((audit / 'audit_checksums.json').read_text(encoding='utf-8')):
        if sha256(audit / r['file']) != r['sha256']:
            raise RuntimeError(f"Audit artifact changed: {r['file']}")
    pairs = csv_in(audit / 'pairs_audit.csv')
    issues = csv_in(audit / 'issues.csv')
    allowed = [r for r in issues if r['type'] == 'pair_integrity' and r['detail'] == 'rgb_depth_spatial_shape_mismatch']
    if len(allowed) != len(issues):
        raise RuntimeError('Unresolved integrity issues; review issues.csv before preprocessing')
    if any((r['rgb_valid'] != 'True' or r['depth_valid'] != 'True' for r in pairs)):
        raise RuntimeError('Unreadable RGB or unusable numeric depth; explicit new resolution required')
    duplicates = csv_in(audit / 'exact_rgb_duplicates.csv')
    if any((r['cross_animal_group'] == 'True' or r['label_conflict'] == 'True' for r in duplicates)):
        raise RuntimeError('Cross-group duplicate or label conflict must be resolved before any split')
    runid = datetime.now().strftime('%Y%m%dT%H%M%S')
    out = project_path('data/processed', ROOT) / ('prepare_' + runid)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'preprocessing_v1.json').write_text(cfgpath.read_text(encoding='utf-8'), encoding='utf-8')
    by_visual = defaultdict(list)
    for r in pairs:
        by_visual[r['episode_id'], r['rgb_decoded_sha256']].append(r)
    pool, mapping = ([], [])
    for _, rows in by_visual.items():
        rows.sort(key=lambda r: (int(r['frame_number']), r['pair_id']))
        representative = rows[0]
        pool.append(representative)
        for r in rows:
            mapping.append({'pair_id': r['pair_id'], 'episode_id': r['episode_id'], 'animal_group': r['animal_group'], 'bcs': r['bcs'], 'representative_pair_id': representative['pair_id'], 'role': 'primary_visual_representative' if r is representative else 'consolidated_same_RGB', 'underlying_depth_sha256': r['depth_sha256'], 'reason': 'same decoded RGB within episode; native depth retained in snapshot', 'structural_resolution': 'independent_letterbox_no_pixel_alignment_claim' if r['shape_match'] == 'False' else 'same_native_array_shape_only'})
    pool.sort(key=lambda r: (r['episode_id'], int(r['frame_number'])))
    for i, r in enumerate(pool):
        r['cache_row'] = i
    csv_out(out / 'all_pairs_to_visual_pool.csv', mapping, list(mapping[0]))
    joint = defaultdict(list)
    for r in pairs:
        joint[r['rgb_sha256'], r['depth_sha256']].append(r)
    joint_rows = []
    for key, rows in joint.items():
        if len(rows) > 1:
            for r in rows:
                joint_rows.append({'pair_id': r['pair_id'], 'rgb_sha256': key[0], 'depth_sha256': key[1], 'group_size': len(rows)})
    csv_out(out / 'exact_joint_duplicate_pairs.csv', joint_rows, ['pair_id', 'rgb_sha256', 'depth_sha256', 'group_size'])
    h, w = cfg['target_size']
    rgb_cache = open_memmap(out / 'rgb_uint8.npy', mode='w+', dtype='uint8', shape=(len(pool), h, w, 3))
    depth_cache = open_memmap(out / 'depth_float16.npy', mode='w+', dtype='float16', shape=(len(pool), h, w))
    mask_cache = open_memmap(out / 'depth_valid_mask.npy', mode='w+', dtype='bool', shape=(len(pool), h, w))
    transforms = []
    for i, r in enumerate(pool):
        rp, dp = (ROOT / r['rgb_relpath'], ROOT / r['depth_relpath'])
        if sha256(rp) != r['rgb_sha256'] or sha256(dp) != r['depth_sha256']:
            raise RuntimeError(f"Snapshot changed since audit: {r['pair_id']}")
        with Image.open(rp) as img:
            rgb = np.asarray(img.convert('RGB'))
        d = np.load(dp, allow_pickle=False).astype('float32')
        valid = np.isfinite(d) & (d > 0)
        lo, hi = np.percentile(d[valid], [1, 99])
        scaled = np.zeros(d.shape, dtype='float32')
        scaled[valid] = np.clip((d[valid] - lo) / (hi - lo), 0, 1)
        rr, rm = letterbox(rgb, (h, w), Image.Resampling.BILINEAR)
        dd, dm = letterbox(scaled, (h, w), Image.Resampling.NEAREST)
        mm, _ = letterbox(valid.astype('uint8'), (h, w), Image.Resampling.NEAREST)
        if not np.isfinite(dd).all() or dd.min() < 0 or dd.max() > 1:
            raise RuntimeError(f"Preprocessing range check failed: {r['pair_id']}")
        rgb_cache[i], depth_cache[i], mask_cache[i] = (rr, dd.astype('float16'), mm.astype(bool))
        tr = {'cache_row': i, 'pair_id': r['pair_id'], 'episode_id': r['episode_id'], 'animal_group': r['animal_group'], 'bcs': r['bcs'], 'outer_fold': r['outer_fold'], 'frame_number': r['frame_number'], 'rgb_relpath': r['rgb_relpath'], 'depth_relpath': r['depth_relpath'], 'rgb_source_sha256': r['rgb_sha256'], 'depth_source_sha256': r['depth_sha256'], 'rgb_tensor_sha256': hashlib.sha256(rr.tobytes()).hexdigest(), 'depth_tensor_sha256': hashlib.sha256(dd.astype('float16').tobytes()).hexdigest(), 'depth_p01': float(lo), 'depth_p99': float(hi), 'native_depth_shape': r['depth_shape']}
        tr.update({'rgb_' + k: v for k, v in rm.items()})
        tr.update({'depth_' + k: v for k, v in dm.items()})
        transforms.append(tr)
        if (i + 1) % 500 == 0:
            print(f'Cached and range-checked {i + 1}/{len(pool)} visual representatives', flush=True)
    rgb_cache.flush()
    depth_cache.flush()
    mask_cache.flush()
    del rgb_cache, depth_cache, mask_cache
    csv_out(out / 'preprocessed_manifest.csv', transforms, list(transforms[0]))
    by_ep = defaultdict(list)
    for r in pool:
        by_ep[r['episode_id']].append(r)
    episodes, slots = ([], [])
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: int(r['frame_number']))
        count = len(rows)
        k = cfg['episode_frames']
        selected = np.linspace(0, count - 1, k).astype(int).tolist() if count >= k else list(range(count)) + [count - 1] * (k - count)
        episodes.append({'episode_id': ep, 'animal_group': rows[0]['animal_group'], 'bcs': rows[0]['bcs'], 'outer_fold': rows[0]['outer_fold'], 'source_pairs': sum((r['episode_id'] == ep for r in pairs)), 'visual_representatives': count, 'evaluation_unique_frames': min(k, count), 'evaluation_slots': k, 'padding_slots': max(0, k - count)})
        for j, idx in enumerate(selected):
            r = rows[idx]
            slots.append({'episode_id': ep, 'slot': j, 'pair_id': r['pair_id'], 'cache_row': r['cache_row'], 'is_padding': j >= count, 'count_as_independent_observation': j < count})
    csv_out(out / 'episodes.csv', episodes, list(episodes[0]))
    csv_out(out / 'evaluation_frame_slots.csv', slots, list(slots[0]))
    splitrows, splitcounts, splitmeta = ([], [], [])
    for outer in [0, 1, 2]:
        (score, valgroups), nval = choose_inner_groups(episodes, outer, cfg['inner_validation'])
        role_rows = []
        for e in episodes:
            role = 'test' if int(e['outer_fold']) == outer else 'validation' if e['animal_group'] in valgroups else 'train'
            row = dict(e, test_fold=outer, role=role)
            role_rows.append(row)
            splitrows.append(row)
        sets = {role: {r['animal_group'] for r in role_rows if r['role'] == role} for role in ['train', 'validation', 'test']}
        assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
        for role in sets:
            rr = [r for r in role_rows if r['role'] == role]
            s = {'test_fold': outer, 'role': role, 'animal_groups': len(sets[role]), 'episodes': len(rr), 'source_pairs': sum((r['source_pairs'] for r in rr)), 'visual_representatives': sum((r['visual_representatives'] for r in rr))}
            s.update({f'BCS{c}_episodes': sum((int(r['bcs']) == c for r in rr)) for c in range(2, 7)})
            splitcounts.append(s)
        splitmeta.append({'outer_test_fold': outer, 'seed': cfg['inner_validation']['seed'] + outer, 'validation_groups': sorted(valgroups), 'selection_score': score, 'group_count': nval, 'uses_test_model_results': False})
    csv_out(out / 'group_splits.csv', splitrows, list(splitrows[0]))
    csv_out(out / 'split_counts.csv', splitcounts, list(splitcounts[0]))
    (out / 'inner_selection.json').write_text(json.dumps(splitmeta, indent=2), encoding='utf-8')
    classcounts = []
    for c in range(2, 7):
        r = [r for r in pairs if int(r['bcs']) == c]
        p = [r for r in pool if int(r['bcs']) == c]
        e = [r for r in episodes if int(r['bcs']) == c]
        classcounts.append({'bcs': c, 'source_pairs': len(r), 'visual_representatives': len(p), 'consolidated_same_rgb': len(r) - len(p), 'episodes': len(e), 'animal_groups': len({x['animal_group'] for x in e}), 'native_depth_480x848': sum((x['depth_shape'] == '480x848' for x in r))})
    csv_out(out / 'class_counts.csv', classcounts, list(classcounts[0]))
    ledger = csv_in(audit / 'stage_ledger.csv')
    ledger += [{'stage': 'R0', 'operation': 'Resolve native-size flags for independent modality encoders', 'unit': 'RGB-D pairs', 'input': len(pairs), 'output': len(pairs), 'excluded': 0, 'evidence': 'all_pairs_to_visual_pool.csv;preprocessing_v1.json', 'status': 'documented_no_pixel_alignment_claim'}, {'stage': 'P0', 'operation': 'Within-episode identical RGB consolidation for primary sampling pool', 'unit': 'visual observations', 'input': len(pairs), 'output': len(pool), 'excluded': len(pairs) - len(pool), 'evidence': 'all_pairs_to_visual_pool.csv', 'status': 'consolidated_source_files_retained'}, {'stage': 'P1', 'operation': 'Letterbox and valid-depth percentile normalisation', 'unit': 'visual observations', 'input': len(pool), 'output': len(pool), 'excluded': 0, 'evidence': 'preprocessed_manifest.csv', 'status': 'measured'}, {'stage': 'P2', 'operation': 'Episode cohort after preprocessing', 'unit': 'episodes / animal groups', 'input': len(pairs), 'output': f"{len(episodes)} / {len({e['animal_group'] for e in episodes})}", 'excluded': 0, 'evidence': 'episodes.csv', 'status': 'no_quality_gating'}, {'stage': 'P3', 'operation': 'Outer test membership with disjoint inner validation groups', 'unit': 'episode-role assignments across 3 rounds', 'input': len(episodes), 'output': len(splitrows), 'excluded': 0, 'evidence': 'group_splits.csv;split_counts.csv', 'status': 'each_episode_tested_once_reused_for_training_in_other_rounds'}]
    csv_out(out / 'stage_ledger.csv', ledger, list(ledger[0]))
    summary = {'run_id': runid, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_audit': audit.relative_to(ROOT).as_posix(), 'snapshot_files': len(pairs) * 2, 'source_pairs': len(pairs), 'resolved_native_size_flags': len(allowed), 'visual_representatives': len(pool), 'consolidated_same_rgb': len(pairs) - len(pool), 'exact_joint_pair_extra_copies': sum((len(r) - 1 for r in joint.values())), 'episodes': len(episodes), 'animal_groups': len({e['animal_group'] for e in episodes}), 'short_episodes_needing_padding': sum((e['padding_slots'] > 0 for e in episodes)), 'distinct_eval_frame_uses': sum((e['evaluation_unique_frames'] for e in episodes)), 'evaluation_tensor_slots': len(slots), 'padding_slots': sum((e['padding_slots'] for e in episodes)), 'training_gate': 'PASS_FOR_EPISODE_LEVEL_NONREGISTERED_INPUTS', 'original_sync_verified': False, 'pixel_registration_verified': False, 'class_counts': classcounts, 'split_counts': splitcounts, 'script_sha256': sha256(Path(__file__)), 'config_sha256': sha256(cfgpath)}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    checks = [{'file': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    (out / 'checksums.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    project_path('provenance/latest_prepared.txt', ROOT).write_text(out.relative_to(ROOT).as_posix(), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
