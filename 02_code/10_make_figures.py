import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))
from project_paths import project_root, project_path, relative_record, get_executable, find_font
os.chdir(project_root())
"""Vector scientific figures from the verified manuscript data, using embedded TNR."""
import json, csv, math, collections
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Rect, Circle, Polygon, String, PolyLine
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib import colors
from reportlab.lib.units import mm
ROOT = project_root()
C = ['#1d5a80', '#b35c35', '#526d2c', '#777777', '#532c72']

def text(d, x, y, s, size=9, anchor='start', bold=False):
    d.add(String(x, y, str(s), fontName='TNRBold' if bold else 'TNR', fontSize=size, textAnchor=anchor, fillColor=colors.black))

def line(d, x1, y1, x2, y2, col='#000000', width=0.7, dash=None):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(col), strokeWidth=width, strokeDashArray=dash))

def box(d, x, y, w, h, lines):
    d.add(Rect(x, y, w, h, fillColor=colors.white, strokeColor=colors.black, strokeWidth=0.7))
    for i, s in enumerate(lines):
        text(d, x + w / 2, y + h - 13 - i * 12, s, 8.5, 'middle', i == 0)

def arrow(d, x1, y, x2):
    line(d, x1, y, x2, y, width=0.8)
    d.add(Polygon([x2, y, x2 - 4, y + 2.3, x2 - 4, y - 2.3], fillColor=colors.black, strokeColor=colors.black))

def marker(d, x, y, col, index=0):
    cc = colors.HexColor(col)
    if index % 3 == 0:
        d.add(Circle(x, y, 2.1, fillColor=colors.white, strokeColor=cc, strokeWidth=0.8))
    elif index % 3 == 1:
        d.add(Rect(x - 2, y - 2, 4, 4, fillColor=colors.white, strokeColor=cc, strokeWidth=0.8))
    else:
        d.add(Polygon([x, y + 2.6, x - 2.5, y - 2, x + 2.5, y - 2], fillColor=colors.white, strokeColor=cc, strokeWidth=0.8))

def axes(d, x, y, w, h, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, panel=None):
    X = lambda z: x + w * (z - xmin) / (xmax - xmin)
    Y = lambda z: y + h * (z - ymin) / (ymax - ymin)
    for t in yticks:
        line(d, x, Y(t), x + w, Y(t), '#dddddd', 0.4)
        text(d, x - 5, Y(t) - 2.5, f'{t:g}', 8, 'end')
    for t in xticks:
        line(d, X(t), y, X(t), y - 3, width=0.6)
        text(d, X(t), y - 13, f'{t:g}', 8, 'middle')
    line(d, x, y, x + w, y)
    line(d, x, y, x, y + h)
    text(d, x + w / 2, y - 28, xlabel, 9, 'middle')
    text(d, x, y + h + 5, ylabel, 9)
    if panel:
        text(d, x - 22, y + h + 5, panel, 9, bold=True)
    return (X, Y)

def save(d, name):
    out = project_path('figures', ROOT)
    out.mkdir(exist_ok=True)
    p = out / (name + '.pdf')
    c = canvas.Canvas(str(p), pagesize=(d.width, d.height), initialFontName='TNR')
    renderPDF.draw(d, c, 0, 0)
    c.save()
    renderSVG.drawToFile(d, str(out / (name + '.svg')))
    print('FIGURE', p, flush=True)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    pdfmetrics.registerFont(TTFont('TNR', find_font(False)))
    pdfmetrics.registerFont(TTFont('TNRBold', find_font(True)))
    renderPDF.STATE_DEFAULTS['fontName'] = 'TNR'
    data = json.loads(project_path('plot_data/research_data.json', ROOT).read_text())
    summary = data['summary']
    pairs = data['paired']
    W = 174 * mm
    get = lambda m, c: next((z for z in summary if z['method'] == m and float(z['budget_cap']) == c))
    d = Drawing(W, 66 * mm)
    bw = 37 * mm
    gap = 8 * mm
    x = 0
    y = 92
    blocks = [['5940 extracted pairs', 'Published aggregate', 'Historical raw bags', 'not re-audited'], ['3815 archived pairs', '7630 native files', 'All copied files', 'SHA256 verified'], ['3177 RGB reps', 'Within-event decoded', 'RGB consolidation', 'No posture filtering'], ['1245 real frames', 'Up to 12 per episode', '151 events retained', 'No padding encoded']]
    for i, b in enumerate(blocks):
        xx = x + i * (bw + gap)
        box(d, xx, y, bw, 65, b)
        if i < 3:
            arrow(d, xx + bw + 2, y + 32, xx + bw + gap - 2)
    text(d, 0, 171, '2125 historical exclusions: published total only', 9)
    counts = data['class_counts']
    text(d, 0, 65, 'Retained-event cohort', 9, bold=True)
    for i, r in enumerate(counts):
        xx = 30 + i * 98
        text(d, xx, 43, 'BCS ' + r['bcs'], 9, 'middle', True)
        text(d, xx, 27, r['episodes'] + ' events', 9, 'middle')
        text(d, xx, 11, r['animal_groups'] + ' identity groups', 8.5, 'middle')
    save(d, 'Fig1')
    d = Drawing(W, 81 * mm)
    text(d, 0, 218, 'Training: leave one animal group out', 9, bold=True)
    bb = [['Fit base scorers', 'Exclude held group', 'Retune on inner', 'validation groups'], ['Held-animal scores', 'RGB and one-frame', 'profiles + action', 'error differences'], ['Fit benefit gate', 'Standardised 8-value', 'score profile', 'No outer-test labels']]
    for i, b in enumerate(bb):
        xx = i * 170
        box(d, xx, 136, 150, 65, b)
        if i < 2:
            arrow(d, xx + 152, 168, xx + 168)
    text(d, 0, 112, 'Inference: charge every probe before choosing refinements', 9, bold=True)
    labels = [['All RGB frames', 'RGB score'], ['One depth probe', '1-frame score'], ['Score profile', 'Budget allocation'], ['Extra depth frames', 'Nested 1/3/6/12'], ['Final event scores', 'RGB fallback allowed']]
    bw = 90
    gap = 10
    for i, b in enumerate(labels):
        xx = i * (bw + gap)
        box(d, xx, 37, bw, 52, b)
        if i < 4:
            arrow(d, xx + bw + 1, 63, xx + bw + gap - 1)
    text(d, 0, 15, 'A probe remains charged even when the final score uses RGB', 9)
    save(d, 'Fig2')
    d = Drawing(W, 94 * mm)
    methods = [('Score_profile_RBF', 'Score profile', C[0], None), ('Uniform_max_affordable', 'Uniform frames', C[1], [5, 2]), ('Binary_strict_crossfit', 'RGB benefit', C[2], [2, 2]), ('Random_affordable_order', 'Random', C[3], [6, 2, 1, 2])]
    for j, (_, lab, col, dash) in enumerate(methods):
        x = j * 124
        line(d, x, 257, x + 17, 257, col, 1, dash)
        text(d, x + 22, 254, lab, 8.5)
    X, Y = axes(d, 31, 40, 202, 184, 0, 1, 0.53, 0.69, [0, 0.25, 0.5, 0.75, 1], [0.55, 0.6, 0.65], 'Depth budget cap', 'MAE (BCS units)', '(a)')
    for j, (m, lab, col, dash) in enumerate(methods):
        rr = sorted([r for r in summary if r['method'] == m], key=lambda r: float(r['budget_cap']))
        pts = [(X(float(r['budget_cap'])), Y(float(r['MAE']))) for r in rr]
        d.add(PolyLine(pts, strokeColor=colors.HexColor(col), strokeWidth=1.05, strokeDashArray=dash))
        for x, y in pts:
            marker(d, x, y, col, j)
    v = float(get('RGBD_all', 1)['MAE'])
    line(d, 31, Y(v), 233, Y(v), '#333333', 0.8, [5, 3])
    text(d, 231, Y(v) + 4, 'Full RGB-D', 8, 'end')
    X, Y = axes(d, 286, 40, 202, 184, 0.5, 1, 0.53, 0.69, [0.5, 0.625, 0.75, 0.875, 1], [0.55, 0.6, 0.65], 'Fraction of full frame calls', 'MAE (BCS units)', '(b)')
    for j, (m, lab, col, dash) in enumerate(methods):
        rr = sorted([r for r in summary if r['method'] == m], key=lambda r: float(r['budget_cap']))
        pts = [(X(1 - float(r['frame_call_reduction'])), Y(float(r['MAE']))) for r in rr]
        d.add(PolyLine(pts, strokeColor=colors.HexColor(col), strokeWidth=1.05, strokeDashArray=dash))
        for x, y in pts:
            marker(d, x, y, col, j)
    marker(d, X(1), Y(v), '#000000')
    save(d, 'Fig3')
    d = Drawing(W, 84 * mm)
    comparators = [('Uniform_max_affordable', 'Uniform frames'), ('RGBD_all', 'Full RGB-D reference'), ('Binary_strict_crossfit', 'RGB benefit'), ('Probe_strict_crossfit', 'High-dimensional probe gate'), ('Score_profile_insample_features', 'In-sample score features'), ('Score_profile_insample_targets', 'In-sample benefit targets'), ('Score_profile_shuffled', 'Shuffled benefit targets'), ('Score_profile_no_count', 'Without frame count')]
    x0 = 191
    w = 285
    y0 = 43
    h = 166
    X = lambda v: x0 + w * (v + 0.08) / 0.12
    for v in [-0.08, -0.04, 0, 0.04]:
        line(d, X(v), y0, X(v), y0 + h, '#aaaaaa' if v == 0 else '#dddddd', 0.7 if v == 0 else 0.4)
        text(d, X(v), y0 - 14, f'{v:.2f}', 8, 'middle')
    for i, (key, label) in enumerate(comparators):
        y = y0 + h - 12 - i * 21
        r = next((z for z in pairs if z['comparator'] == key and z['endpoint'] == 'mean_caps25_50_75'))
        lo = float(r['paired_cluster_CI_low'])
        hi = float(r['paired_cluster_CI_high'])
        v = float(r['delta_MAE'])
        text(d, x0 - 9, y - 3, label, 8.5, 'end')
        line(d, X(lo), y, X(hi), y, C[0], 1)
        line(d, X(lo), y - 3, X(lo), y + 3, C[0])
        line(d, X(hi), y - 3, X(hi), y + 3, C[0])
        marker(d, X(v), y, C[0])
    line(d, x0, y0, x0 + w, y0)
    text(d, x0 + w / 2, 12, 'Delta MAE: score profile minus comparator', 9, 'middle')
    save(d, 'Fig4')
    d = Drawing(W, 96 * mm)
    stress = [('nominal', 'Original'), ('random_drop_10', 'Drop10%'), ('random_drop_30', 'Drop30%'), ('central_hole_25', 'Block25%')]
    X, Y = axes(d, 31, 42, 205, 181, 0, 3, 0.54, 0.69, [], [0.55, 0.6, 0.65], 'Depth input condition', 'MAE (BCS units)', '(a)')
    for j, (m, lab, col) in enumerate([('Score_profile_RBF', 'Score profile', C[0]), ('Uniform_max_affordable', 'Uniform', C[1]), ('RGBD_all', 'Full RGB-D', C[3])]):
        vv = [float(get(m, 0.25)['MAE'])] + [float(next((z for z in data['stress'] if z['method'] == m and z['profile'] == s and (z['budget_cap'] == 0.25)))['MAE']) for s, l in stress[1:]]
        d.add(PolyLine([(X(i), Y(v)) for i, v in enumerate(vv)], strokeColor=colors.HexColor(col), strokeWidth=1.05, strokeDashArray=[4, 2] if j else None))
        for i, v in enumerate(vv):
            marker(d, X(i), Y(v), col, j)
        line(d, 31 + j * 77, 253, 46 + j * 77, 253, col, 1, [4, 2] if j else None)
        text(d, 50 + j * 77, 250, lab, 8)
    for i, (_, lab) in enumerate(stress):
        text(d, X(i), 29, lab, 8, 'middle')
    x0 = 363
    w = 121
    maxsec = 30
    xx = lambda v: x0 + w * v / maxsec
    text(d, 277, 228, '(b)', 9, bold=True)
    text(d, 335, 228, 'Native-file batch time', 9)
    tt = [('full', 1, 'Full RGB-D'), ('uniform', 0.5, 'Uniform .50'), ('profile', 0.5, 'Profile .50'), ('uniform', 0.25, 'Uniform .25'), ('profile', 0.25, 'Profile .25'), ('RGB', 0, 'RGB only')]
    for i, (policy, cap, label) in enumerate(tt):
        r = next((z for z in data['timing'] if z['policy'] == policy and z['budget_cap'] == cap))
        y = 207 - i * 26
        text(d, x0 - 6, y - 3, label, 8.5, 'end')
        d.add(Rect(x0, y - 6, xx(r['median_seconds']) - x0, 12, fillColor=colors.HexColor(C[0] if policy == 'profile' else '#bbbbbb'), strokeColor=colors.black, strokeWidth=0.5))
        line(d, xx(r['minimum_seconds']), y, xx(r['maximum_seconds']), y)
        line(d, xx(r['minimum_seconds']), y - 3, xx(r['minimum_seconds']), y + 3)
        line(d, xx(r['maximum_seconds']), y - 3, xx(r['maximum_seconds']), y + 3)
    line(d, x0, 42, x0 + w, 42)
    for v in [0, 10, 20, 30]:
        line(d, xx(v), 42, xx(v), 39)
        text(d, xx(v), 27, str(v), 8, 'middle')
    text(d, x0 + w / 2, 12, 'Seconds per 51-event batch', 9, 'middle')
    save(d, 'Fig5')
    d = Drawing(W, 78 * mm)
    X, Y = axes(d, 34, 41, 446, 143, 1.5, 6.5, 0, 3, [2, 3, 4, 5, 6], [0, 1, 2, 3], 'Reference BCS', 'Class-specific MAE')
    for j, (m, label, col) in enumerate([('Score_profile_RBF', 'Score profile', C[0]), ('Uniform_max_affordable', 'Uniform', C[1]), ('RGBD_all', 'Full RGB-D', C[3])]):
        for c in range(2, 7):
            r = next((z for z in data['per_class'] if z['method'] == m and float(z['budget_cap']) == 0.25 and (float(z['BCS']) == c)))
            x = X(c) + (j - 1) * 16
            h = Y(float(r['MAE'])) - Y(0)
            d.add(Rect(x - 7, Y(0), 14, h, fillColor=colors.HexColor(col), strokeColor=colors.black, strokeWidth=0.4))
        line(d, 34 + j * 148, 207, 51 + j * 148, 207, col, 2)
        text(d, 57 + j * 148, 204, label, 8.5)
    save(d, 'FigS1')
    analysis = ROOT / data['sources']['analysis']
    allrows = read(analysis / 'all_test_predictions.csv')
    vals = collections.defaultdict(list)
    for r in allrows:
        if r['method'] == 'Score_profile_RBF' and float(r['budget_cap']) == 0.25:
            vals[r['episode_id'], float(r['reference'])].append(float(r['prediction']))
    d = Drawing(84 * mm, 86 * mm)
    X, Y = axes(d, 31, 41, 194, 178, 1.8, 6.2, 1.8, 6.2, [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], 'Reference BCS', 'Mean predicted BCS')
    line(d, X(2), Y(2), X(6), Y(6), '#777777', 0.7, [4, 2])
    for (ep, y), v in vals.items():
        marker(d, X(y), Y(sum(v) / len(v)), C[0])
    save(d, 'FigS2')
    d = Drawing(W, 70 * mm)
    X, Y = axes(d, 34, 40, 446, 130, -0.5, 4.5, 0, 100, [], [0, 25, 50, 75, 100], 'Chosen depth-frame level', 'Mean number of events')
    levels = [0, 1, 3, 6, 12]
    for j, cap in enumerate([0.25, 0.5]):
        count = collections.Counter((int(r['depth_level']) for r in allrows if r['method'] == 'Score_profile_RBF' and float(r['budget_cap']) == cap))
        for i, k in enumerate(levels):
            d.add(Rect(X(i) + (j - 0.5) * 19 - 8, Y(0), 16, Y(count[k] / 3) - Y(0), fillColor=colors.HexColor(C[j]), strokeColor=colors.black, strokeWidth=0.5))
        line(d, 34 + j * 180, 188, 51 + j * 180, 188, C[j], 2)
        text(d, 57 + j * 180, 185, 'Depth cap ' + str(cap), 8.5)
    for i, k in enumerate(levels):
        text(d, X(i), 27, str(k), 8, 'middle')
    save(d, 'FigS3')
if __name__ == '__main__':
    main()
