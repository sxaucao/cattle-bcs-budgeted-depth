from project_paths import project_root, project_path, relative_record, get_executable, find_font
"""Matched partially fine-tuned episode experiments with explicit train/val/test roles.

Smoke mode never evaluates the outer test set. Formal configuration is copied
before training and every failed/completed attempt remains in a separate folder.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, csv, json, math, random, time, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone
from importlib import import_module
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
u = import_module('01_audit_and_snapshot')
ROOT = project_root()

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EpisodeModel(nn.Module):

    def __init__(self, modalities, layer4_state):
        super().__init__()
        self.modalities = modalities
        self.branches = nn.ModuleDict()
        for mod in modalities:
            b = resnet18(weights=None).layer4
            b.load_state_dict(layer4_state)
            self.branches[mod] = b
        self.head = nn.Sequential(nn.Linear(512 * len(modalities), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, arrays, bag_sizes):
        feats = []
        for mod in self.modalities:
            h = self.branches[mod](arrays[mod]).mean(dim=(-2, -1))
            feats.append(torch.stack([x.mean(0) for x in h.split(bag_sizes)]))
        return self.head(torch.cat(feats, dim=1))

class Bundle:

    def __init__(self, feature_dir, cfg):
        self.feature_dir = feature_dir
        self.cfg = cfg
        self.prepared = ROOT / json.loads((feature_dir / 'summary.json').read_text())['prepared']
        self.cache = {m: np.load(feature_dir / (m + '_stage3.npy'), mmap_mode='r') for m in ['rgb', 'depth']}
        rows = u.csv_in(self.prepared / 'preprocessed_manifest.csv')
        self.by_ep = {}
        self.native = {}
        for r in rows:
            self.by_ep.setdefault(r['episode_id'], []).append(r)
        for ep, rs in self.by_ep.items():
            rs.sort(key=lambda r: int(r['frame_number']))
            self.native[ep] = int(np.mean([r['native_depth_shape'] == '480x848' for r in rs]) >= 0.5)
        self.splits = u.csv_in(self.prepared / 'group_splits.csv')

    def split(self, fold, role):
        return sorted([r for r in self.splits if int(r['test_fold']) == fold and r['role'] == role], key=lambda r: r['episode_id'])

    def batch(self, episodes, modalities, device, training=False, seed=0):
        selected = []
        sizes = []
        views = []
        for e in episodes:
            ep = e['episode_id']
            rows = self.by_ep[ep]
            n = len(rows)
            k = min(n, self.cfg['episode_frames'])
            if training:
                es = int(hashlib.sha256(ep.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng((seed + es) % 2 ** 32)
                idx = np.sort(rng.choice(n, k, replace=False))
                view = int(rng.integers(2))
            else:
                idx = np.linspace(0, n - 1, k).astype(int)
                view = 0
            selected.extend((int(rows[j]['cache_row']) for j in idx))
            views.extend([view] * k)
            sizes.append(k)
        selected = np.array(selected)
        views = np.array(views)
        arrays = {m: torch.from_numpy(np.array(self.cache[m][selected, views], copy=True)).to(device=device, dtype=torch.float32) for m in modalities}
        y = torch.tensor([int(e['bcs']) - 2 for e in episodes], device=device, dtype=torch.long)
        return (arrays, sizes, y)

def metrics(logits, y):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    pred = probs.argmax(1)
    y = np.asarray(y, dtype=int)
    expected = probs @ np.arange(2, 7)
    qwk = cohen_kappa_score(y, pred, labels=list(range(5)), weights='quadratic')
    return {'expected_MAE': float(np.abs(expected - (y + 2)).mean()), 'argmax_MAE': float(np.abs(pred - y).mean()), 'accuracy': float(accuracy_score(y, pred)), 'macro_F1': float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0)), 'QWK': float(qwk), 'large_error_ge_2': float((abs(pred - y) >= 2).mean()), 'within_one': float((abs(pred - y) <= 1).mean()), 'CE': float(F.cross_entropy(torch.tensor(logits), torch.tensor(y)).item()), 'n_episodes': len(y), 'coverage': 1.0}

def evaluate(model, bundle, episodes, device, batch=4):
    model.eval()
    ls = []
    ys = []
    with torch.inference_mode():
        for st in range(0, len(episodes), batch):
            arr, sizes, y = bundle.batch(episodes[st:st + batch], model.modalities, device)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                l = model(arr, sizes)
            ls.append(l.float().cpu().numpy())
            ys.extend(y.cpu().numpy().tolist())
    logits = np.concatenate(ls)
    return (logits, np.array(ys), metrics(logits, ys))

def cdf_gt(y):
    return (y[:, None] > torch.arange(4, device=y.device)[None]).float()

def q_survival(p):
    return 1 - torch.cumsum(p, dim=1)[:, :-1]

def load_checkpoint(path, layer4, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = EpisodeModel(ck['modalities'], layer4)
    model.load_state_dict(ck['model_state'])
    return model.to(device)

def teacher_targets(run, fold, seed, bundle, episodes, layer4, device, variant, cfg, attempt):
    targets = {}
    required = ['RGB_CE'] if variant == 'RGB_self_KD' else ['RGB_CE', 'RGBD_CE'] if variant == 'RGB_rgbd_KD' else ['RGB_CE', 'Depth_CE']
    for arm in required:
        parent = run / f'fold{fold}_seed{seed}' / arm
        done = json.loads((parent / 'completion.json').read_text())
        model = load_checkpoint(run / done['checkpoint'], layer4, device)
        logits, _, _ = evaluate(model, bundle, episodes, device)
        targets[arm] = torch.softmax(torch.tensor(logits) / cfg['temperature'], dim=1)
        del model
    if variant == 'RGB_self_KD':
        return (targets['RGB_CE'], None)
    pt = targets['RGBD_CE'] if variant == 'RGB_rgbd_KD' else targets['Depth_CE']
    pr = targets['RGB_CE']
    mapping = []
    if variant == 'RGB_shuffled_depth_benefit':
        rng = np.random.default_rng(seed + fold * 1009 + 678)
        chosen = []
        for i, e in enumerate(episodes):
            candidates = [j for j, a in enumerate(episodes) if a['animal_group'] != e['animal_group'] and bundle.native[a['episode_id']] == bundle.native[e['episode_id']]]
            fallback = False
            if not candidates:
                candidates = [j for j, a in enumerate(episodes) if a['animal_group'] != e['animal_group']]
                fallback = True
            j = int(rng.choice(candidates))
            chosen.append(j)
            mapping.append({'target_episode': e['episode_id'], 'source_depth_episode': episodes[j]['episode_id'], 'target_animal': e['animal_group'], 'source_animal': episodes[j]['animal_group'], 'target_width_stratum': bundle.native[e['episode_id']], 'source_width_stratum': bundle.native[episodes[j]['episode_id']], 'stratum_fallback': fallback})
        pt = pt[chosen]
        u.csv_out(attempt / 'training_depth_mismatch.csv', mapping, list(mapping[0]))
    y = torch.tensor([int(e['bcs']) - 2 for e in episodes])
    z = cdf_gt(y)
    w = torch.clamp((q_survival(pr) - z) ** 2 - (q_survival(pt) - z) ** 2, min=0)
    positive = w[w > 0]
    w = torch.clamp(w / (positive.mean() if len(positive) else 1), 0, 4)
    np.savez_compressed(attempt / 'training_teacher_targets.npz', episode_ids=np.array([e['episode_id'] for e in episodes]), depth_p=pt.numpy(), rgb_p=pr.numpy(), boundary_weights=w.numpy())
    return (pt, w)

def write_predictions(path, episodes, logits, role, fold, seed, variant):
    p = F.softmax(torch.tensor(logits), dim=1).numpy()
    rows = []
    for e, pr in zip(episodes, p):
        rows.append({'variant': variant, 'fold': fold, 'seed': seed, 'role': role, 'episode_id': e['episode_id'], 'animal_group': e['animal_group'], 'reference_BCS': int(e['bcs']), 'predicted_BCS': int(pr.argmax()) + 2, 'expected_BCS': float(pr @ np.arange(2, 7)), **{f'p_BCS{c + 2}': float(pr[c]) for c in range(5)}})
    u.csv_out(path, rows, list(rows[0]))

def train_one(run, fold, seed, variant, bundle, cfg, layer4, device, mode):
    parent = run / f'fold{fold}_seed{seed}' / variant
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / 'completion.json').exists():
        print(f'SKIP complete {fold} {seed} {variant}', flush=True)
        return
    attempt = parent / ('attempt_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    attempt.mkdir(exist_ok=False)
    tr = bundle.split(fold, 'train')
    va = bundle.split(fold, 'validation')
    te = bundle.split(fold, 'test')
    groups = [{e['animal_group'] for e in rr} for rr in [tr, va, te]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    u.csv_out(attempt / 'episode_roles.csv', [dict(e) for e in tr + va + te], list(tr[0]))
    targets = weights_boundary = None
    prototype_variants = ['RGB_gaussian', 'RGB_rgb_prototype', 'RGB_rgbd_prototype', 'RGB_entropy_matched', 'RGB_mismatched_prototype']
    if variant in prototype_variants:
        targets = import_module('09_ordinal_prototype_targets').build(run, fold, seed, bundle, tr, layer4, device, variant, cfg, attempt)
    elif variant not in ['RGB_CE', 'Depth_CE', 'RGBD_CE', 'RGB_ordinal']:
        targets, weights_boundary = teacher_targets(run, fold, seed, bundle, tr, layer4, device, variant, cfg, attempt)
    setseed(seed + fold * 103)
    mods = ['depth'] if variant == 'Depth_CE' else ['rgb', 'depth'] if variant == 'RGBD_CE' else ['rgb']
    model = EpisodeModel(mods, layer4).to(device)
    opt = torch.optim.AdamW([{'params': model.branches.parameters(), 'lr': cfg['learning_rate_layer4']}, {'params': model.head.parameters(), 'lr': cfg['learning_rate_head']}], weight_decay=cfg['weight_decay'])
    epochs = 3 if mode == 'smoke' else cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-06)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    counts = np.bincount([int(e['bcs']) - 2 for e in tr], minlength=5)
    cw = 1 / np.sqrt(np.maximum(counts, 1))
    cw = cw / cw.mean()
    cw = torch.tensor(cw, dtype=torch.float32, device=device)
    lossfn = nn.CrossEntropyLoss(weight=cw)
    batch = cfg['batch_episodes']
    best = (float('inf'), float('inf'))
    stale = 0
    log = []
    start = time.perf_counter()
    tidx = {e['episode_id']: i for i, e in enumerate(tr)}
    print(f'START fold={fold} seed={seed} variant={variant} train={len(tr)} val={len(va)}', flush=True)
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + fold * 103 + epoch * 7919)
        order = rng.permutation(len(tr))
        losses = []
        for st in range(0, len(order), batch):
            ix = order[st:st + batch]
            eps = [tr[i] for i in ix]
            arr, sizes, y = bundle.batch(eps, mods, device, training=True, seed=seed + fold * 103 + epoch * 7919)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = model(arr, sizes)
                if variant in prototype_variants:
                    pt = targets[ix].to(device)
                    per = -(pt * F.log_softmax(logits.float(), dim=1)).sum(1)
                    loss = (per * cw[y]).sum() / cw[y].sum()
                else:
                    loss = lossfn(logits.float(), y)
                ordinal = variant in ['RGB_ordinal', 'RGB_depth_ordinal', 'RGB_depth_benefit', 'RGB_shuffled_depth_benefit', 'RGB_selective_ordinal']
                if ordinal:
                    loss = loss + cfg['hard_ordinal_weight'] * F.mse_loss(q_survival(F.softmax(logits.float(), dim=1)), cdf_gt(y))
                if targets is not None and variant not in prototype_variants:
                    pt = targets[ix].to(device)
                    temp = cfg['temperature']
                    if variant in ['RGB_self_KD', 'RGB_depth_KD', 'RGB_rgbd_KD', 'RGB_selective_KD']:
                        per = F.kl_div(F.log_softmax(logits.float() / temp, dim=1), pt, reduction='none').sum(1)
                        if variant == 'RGB_selective_KD':
                            per = per * (pt.argmax(1) == y).float()
                        kd = per.mean() * temp * temp
                    else:
                        sq = (q_survival(F.softmax(logits.float() / temp, dim=1)) - q_survival(pt)) ** 2
                        if variant in ['RGB_depth_benefit', 'RGB_shuffled_depth_benefit']:
                            sq = sq * weights_boundary[ix].to(device)
                        if variant == 'RGB_selective_ordinal':
                            sq = sq * (pt.argmax(1) == y).float()[:, None]
                        kd = sq.mean() * temp * temp
                    loss = loss + cfg['distillation_weight'] * kd
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        vl, vy, vm = evaluate(model, bundle, va, device, batch)
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **{'val_' + k: v for k, v in vm.items()}, 'elapsed_seconds': time.perf_counter() - start}
        log.append(row)
        u.csv_out(attempt / 'learning_curve.csv', log, list(row))
        key = (vm['expected_MAE'], vm['CE'])
        if key < best:
            best = key
            stale = 0
            torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'modalities': mods, 'variant': variant, 'fold': fold, 'seed': seed, 'epoch': epoch + 1, 'validation': vm, 'experiment_sha256': u.sha256(run / 'experiment.json')}, attempt / 'best.pt')
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  {variant} f{fold} s{seed} epoch={epoch + 1} val_MAE={vm['expected_MAE']:.4f} best={best[0]:.4f} time={time.perf_counter() - start:.1f}s", flush=True)
        if mode != 'smoke' and epoch + 1 >= cfg['minimum_epochs'] and (stale >= cfg['early_stopping_patience']):
            break
    del model
    model = load_checkpoint(attempt / 'best.pt', layer4, device)
    outcomes = {}
    for role, eps in [('train', tr), ('validation', va)] + ([] if mode == 'smoke' else [('test', te)]):
        logits, y, mm = evaluate(model, bundle, eps, device, batch)
        outcomes[role] = mm
        write_predictions(attempt / (role + '_predictions.csv'), eps, logits, role, fold, seed, variant)
    done = {'variant': variant, 'fold': fold, 'seed': seed, 'mode': mode, 'checkpoint': (attempt / 'best.pt').relative_to(run).as_posix(), 'attempt': attempt.relative_to(run).as_posix(), 'epochs_trained': len(log), 'runtime_seconds': time.perf_counter() - start, 'metrics': outcomes, 'train_class_counts': counts.tolist(), 'script_sha256': u.sha256(Path(__file__)), 'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0, 'test_used_for_selection': False, 'teacher_targets_generated_on': 'training episodes only; in-sample teachers, not cross-fitted' if targets is not None else None}
    (attempt / 'run_info.json').write_text(json.dumps(done, indent=2), encoding='utf-8')
    (attempt / 'checksums.json').write_text(json.dumps([{'file': p.name, 'sha256': u.sha256(p)} for p in sorted(attempt.iterdir()) if p.is_file()], indent=2), encoding='utf-8')
    (parent / 'completion.json').write_text(json.dumps(done, indent=2), encoding='utf-8')
    print('DONE ' + json.dumps({'variant': variant, 'fold': fold, 'seed': seed, 'seconds': done['runtime_seconds'], 'metrics': outcomes}), flush=True)
    del model, opt
    if device == 'cuda':
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'formal'], default='smoke')
    parser.add_argument('--run-dir')
    parser.add_argument('--config', default='configs/formal_experiment_v1.json')
    a = parser.parse_args()
    import_module('00_archive_sources').archive()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfgpath = ROOT / a.config
    cfg = json.loads(cfgpath.read_text())
    features = ROOT / project_path('provenance/latest_features.txt', ROOT).read_text().strip()
    for r in json.loads((features / 'checksums.json').read_text()):
        if u.sha256(features / r['file']) != r['sha256']:
            raise RuntimeError('Feature-cache checksum mismatch')
    run = Path(a.run_dir) if a.run_dir else project_path('runs', ROOT) / (a.mode + '_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'experiment.json').exists():
        if json.loads((run / 'experiment.json').read_text()) != cfg:
            raise RuntimeError('Refusing changed configuration on an existing run')
    else:
        (run / 'experiment.json').write_text(cfgpath.read_text(), encoding='utf-8')
    if not (run / 'execution.json').exists():
        (run / 'execution.json').write_text(json.dumps({'mode': a.mode, 'started_utc': datetime.now(timezone.utc).isoformat(), 'features': features.relative_to(ROOT).as_posix(), 'script_sha256': u.sha256(Path(__file__)), 'torch': torch.__version__, 'python': sys.version, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'test_selection': False}, indent=2), encoding='utf-8')
    bundle = Bundle(features, cfg)
    base = resnet18(weights=None)
    base.load_state_dict(torch.load(cfg['pretrained_weights'], map_location='cpu', weights_only=True))
    l4 = copy.deepcopy(base.layer4.state_dict())
    del base
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folds = [0] if a.mode == 'smoke' else cfg['outer_folds']
    seeds = cfg['seeds'][:1] if a.mode == 'smoke' else cfg['seeds']
    variants = ['RGB_CE', 'Depth_CE'] if a.mode == 'smoke' else cfg['variants']
    (project_path('provenance', ROOT) / ('latest_' + a.mode + '_experiment.txt')).write_text(run.relative_to(ROOT).as_posix(), encoding='utf-8')
    for fold in folds:
        for seed in seeds:
            for v in variants:
                train_one(run, fold, seed, v, bundle, cfg, l4, device, a.mode)
    (run / 'COMPLETE.json').write_text(json.dumps({'finished_utc': datetime.now(timezone.utc).isoformat(), 'mode': a.mode, 'trained_models': len(folds) * len(seeds) * len(variants)}, indent=2), encoding='utf-8')
    print('RUN COMPLETE ' + str(run), flush=True)
