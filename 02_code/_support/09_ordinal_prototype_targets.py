from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Training-only reference-anchored prototype targets with animal-excluded means."""
import json, math
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.nn import functional as F

def representation(model, bundle, episodes, device):
    all_features = []
    model.eval()
    with torch.inference_mode():
        for st in range(0, len(episodes), 4):
            arr, sizes, y = bundle.batch(episodes[st:st + 4], model.modalities, device)
            fs = []
            for mod in model.modalities:
                h = model.branches[mod](arr[mod]).mean((-2, -1))
                f = torch.stack([z.mean(0) for z in h.split(sizes)])
                fs.append(F.normalize(f, dim=1))
            all_features.append(F.normalize(torch.cat(fs, dim=1), dim=1).cpu().numpy())
    return np.concatenate(all_features)

def gaussian(y, sigma):
    return torch.softmax(-(torch.arange(5)[None] - y[:, None]) ** 2 / (2 * sigma * sigma), dim=1)

def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-10))).sum(1)

def build(run, fold, seed, bundle, episodes, layer4, device, variant, cfg, attempt):
    core = import_module('07_train_experiments')
    u = core.u
    y = torch.tensor([int(e['bcs']) - 2 for e in episodes])
    g = gaussian(y, cfg['prototype_label_sigma'])
    if variant == 'RGB_gaussian':
        target = g
    else:
        teacher_arm = 'RGB_CE' if variant == 'RGB_rgb_prototype' else 'RGBD_CE'
        done = json.loads((run / f'fold{fold}_seed{seed}' / teacher_arm / 'completion.json').read_text())
        model = core.load_checkpoint(run / done['checkpoint'], layer4, device)
        true_f = representation(model, bundle, episodes, device)
        del model
        f = true_f.copy()
        mapping = []
        if variant == 'RGB_mismatched_prototype':
            rng = np.random.default_rng(seed + fold * 1009 + 2017)
            for i, e in enumerate(episodes):
                js = [j for j, a in enumerate(episodes) if a['animal_group'] != e['animal_group'] and bundle.native[a['episode_id']] == bundle.native[e['episode_id']]]
                fallback = False
                if not js:
                    js = [j for j, a in enumerate(episodes) if a['animal_group'] != e['animal_group']]
                    fallback = True
                j = int(rng.choice(js))
                f[i, 512:] = true_f[j, 512:]
                mapping.append({'target_episode': e['episode_id'], 'source_depth_episode': episodes[j]['episode_id'], 'same_native_width_stratum': bundle.native[e['episode_id']] == bundle.native[episodes[j]['episode_id']], 'fallback': fallback})
            f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-08)
            u.csv_out(attempt / 'prototype_mismatch_mapping.csv', mapping, list(mapping[0]))
        scores = []
        audit = []
        yn = y.numpy()
        for i, e in enumerate(episodes):
            mu = []
            for c in range(5):
                js = [j for j, a in enumerate(episodes) if yn[j] == c and a['animal_group'] != e['animal_group']]
                fallback = not js
                if fallback:
                    js = np.where(yn == c)[0].tolist()
                if not js:
                    raise RuntimeError('Class has no prototype support')
                m = true_f[js].mean(0)
                m = m / max(np.linalg.norm(m), 1e-08)
                mu.append(m)
                audit.append({'target_episode': e['episode_id'], 'target_animal': e['animal_group'], 'prototype_BCS': c + 2, 'support_episodes': len(js), 'support_animal_groups': len({episodes[j]['animal_group'] for j in js}), 'animal_exclusion_fallback': fallback})
            scores.append(np.array(mu) @ f[i])
        scores = torch.tensor(np.stack(scores), dtype=torch.float32)
        logtarget = torch.log(g.clamp_min(1e-12)) + scores / cfg['prototype_similarity_temperature']
        target = torch.softmax(logtarget, dim=1)
        u.csv_out(attempt / 'prototype_support_audit.csv', audit, list(audit[0]))
        if variant == 'RGB_entropy_matched':
            desired = entropy(target)
            lo = torch.full((len(y),), 0.001)
            hi = torch.full((len(y),), 10.0)
            for _ in range(50):
                mid = (lo + hi) / 2
                q = torch.softmax(-(torch.arange(5)[None] - y[:, None]) ** 2 / (2 * mid[:, None] ** 2), dim=1)
                h = entropy(q)
                lo = torch.where(h < desired, mid, lo)
                hi = torch.where(h >= desired, mid, hi)
            sigma = (lo + hi) / 2
            target = torch.softmax(-(torch.arange(5)[None] - y[:, None]) ** 2 / (2 * sigma[:, None] ** 2), dim=1)
            np.save(attempt / 'entropy_matched_sigmas.npy', sigma.numpy())
    np.savez_compressed(attempt / 'ordinal_soft_targets.npz', episode_ids=np.array([e['episode_id'] for e in episodes]), target=target.numpy(), gaussian=g.numpy(), reference_BCS=y.numpy() + 2)
    return target
