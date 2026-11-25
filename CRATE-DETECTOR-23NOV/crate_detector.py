"""
Crate defect detector - one file, all models, pick with MODE.

    from crate_detector import CrateDetector
    det = CrateDetector("/Workspace/.../CRATE-DETECTOR-23NOV", mode="single")   # or "ensemble", "bigdefect", "single_bigdefect"
    out = det.predict("crate.png")       # path, PIL image, or RGB numpy array
    out["crate_flagged"]                 # True if any defect >= threshold
    out["defects"]                       # every box: [{"class": "Crack", "score": 0.91, "box_xyxy": [x0, y0, x1, y1]}, ...]
    out["damage_areas"]                  # ONE box + ONE label per damaged area (overlapping/nearby boxes merged) - show these
    det.draw("crate.png", out, "annotated.jpg")   # annotated image (damage_areas by default)
    det.report(out, frame_id="crate.png")         # verdict JSON like the pipeline diagram (DAMAGE_DETECTED / NO_DEFECT_DETECTED)
All 6 classes are reported. To hide a class (e.g. water damage): config.json "drop_classes": ["Water damage"] or CrateDetector(..., drop_classes=["Water damage"]).

Modes (all recall numbers on our validation set, threshold set so <= 6 of 36 healthy views get any box):
  single            1 model  (current model)              all 36.8% | big(>=32px) 50.8% |  ~45 ms
  ensemble          4 models (best on ALL sizes)          all 41.7% | big 56.7%         | ~150 ms
  bigdefect         4 models (best on BIG defects)        all 40.8% | big 67.8%         | ~150 ms
  single_bigdefect  1 model  (big-defect model alone)     all 32.3% | big 56.7%         |  ~45 ms
Needs: torch, torchvision, numpy, pillow  (+ ultralytics only for mode "ensemble").
"""
import json, os, collections
import numpy as np
import torch
from PIL import Image

CLASSES = ['Crack', 'Dent', 'Water damage', 'Scratch', 'Hole', 'Broken edge']


def _build_frcnn(spec):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.rpn import RPNHead
    tr, te = map(int, spec['topn'].split(','))
    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=1024, max_size=1024,
                                   box_detections_per_img=spec['dets'], box_score_thresh=0.001, box_nms_thresh=0.7,
                                   rpn_pre_nms_top_n_train=tr, rpn_post_nms_top_n_train=tr, rpn_pre_nms_top_n_test=te, rpn_post_nms_top_n_test=te)
    sizes = [tuple(int(s) for s in lvl.split('/')) for lvl in spec['sizes'].split(',')]
    ratios = tuple(float(r) for r in spec['ratios'].split(','))
    m.rpn.anchor_generator = AnchorGenerator(sizes=tuple(sizes), aspect_ratios=(ratios,) * len(sizes))
    na = len(sizes[0]) * len(ratios)
    if na != 3: m.rpn.head = RPNHead(256, na, conv_depth=2)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return m


def _iou_xywh(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0., min(ax1, bx1) - max(a[0], b[0])), max(0., min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.


def _wbf(per_model, n_models, iou=0.55, topk=100, min_score=0.01):
    """Weighted box fusion, same as used for the scores: boxes of the same class with IoU >= .55 are merged (score-weighted
    box), fused score = sum over models of that model's best score in the cluster / number of models."""
    by = collections.defaultdict(list)
    for name, dets in per_model.items():
        for d in dets:
            if d['score'] >= min_score: by[d['category_id']].append((d['score'], d['bbox'], name))
    rows = []
    for c, L in by.items():
        L.sort(key=lambda x: -x[0]); clusters = []
        for s, b, n in L:
            best, bi = 0, -1
            for i, cl in enumerate(clusters):
                v = _iou_xywh(cl['box'], b)
                if v > best: best, bi = v, i
            if best >= iou:
                cl = clusters[bi]; cl['m'].append((s, b, n)); w = np.array([x[0] for x in cl['m']])
                cl['box'] = list((np.array([x[1] for x in cl['m']]) * w[:, None]).sum(0) / w.sum())
            else: clusters.append({'box': list(b), 'm': [(s, b, n)]})
        for cl in clusters:
            per = {}
            for s, b, n in cl['m']: per[n] = max(per.get(n, 0), s)
            rows.append({'category_id': c, 'bbox': [float(v) for v in cl['box']], 'score': float(sum(per.values()) / n_models)})
    return sorted(rows, key=lambda x: -x['score'])[:topk]


def _weight_path(folder, fname, cache=None):
    """Path of a model file. Uses models/<fname> if it exists. Otherwise the file is stored in git as parts
    (models/<fname>.part00, .part01 ... - GitHub refuses files > 100 MB): they are joined ONCE, byte for byte, into a cache
    folder on local disk (default: <temp dir>/crate_detector_weights, e.g. /tmp on Databricks), never into the package folder."""
    import glob, tempfile
    whole = os.path.join(folder, 'models', fname)
    if os.path.exists(whole): return whole
    parts = sorted(glob.glob(whole + '.part*'))
    if not parts: raise FileNotFoundError(f'{whole} not found and no {fname}.part* files - see GUIDE.txt')
    cache = cache or os.path.join(tempfile.gettempdir(), 'crate_detector_weights'); os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, fname)
    if not os.path.exists(out) or os.path.getsize(out) != sum(os.path.getsize(p) for p in parts):
        tmp = out + '.joining'
        with open(tmp, 'wb') as o:
            for p in parts:
                with open(p, 'rb') as f:
                    for b in iter(lambda: f.read(1 << 24), b''): o.write(b)
        os.replace(tmp, out)
    return out


def merge_areas(defects, img_w, img_h, gap=30, max_frac=0.25):
    """One box + one label per damaged area (for display / reporting).
    Boxes that overlap or are within `gap` px of each other are the same damaged area; the area box surrounds all of them and
    takes the label + score of its most confident box. Very large boxes (> max_frac of the photo) are never merged into others."""
    big = lambda b: (b[2] - b[0]) * (b[3] - b[1]) > max_frac * img_w * img_h
    near = lambda a, b: not (big(a) or big(b)) and not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])
    groups = []
    for d in sorted(defects, key=lambda d: -d['score']):
        hit = [g for g in groups if any(near(d['box_xyxy'], x['box_xyxy']) for x in g)]
        if not hit: groups.append([d]); continue
        hit[0].append(d)
        for g in hit[1:]: hit[0].extend(g); groups.remove(g)
    out = []
    for g in groups:
        top = max(g, key=lambda x: x['score'])
        out.append({'class': top['class'], 'score': top['score'], 'n_boxes': len(g),
                    'box_xyxy': [min(x['box_xyxy'][0] for x in g), min(x['box_xyxy'][1] for x in g), max(x['box_xyxy'][2] for x in g), max(x['box_xyxy'][3] for x in g)]})
    return sorted(out, key=lambda d: -d['score'])


COLORS = {'Crack': (255, 40, 40), 'Broken edge': (255, 150, 0), 'Hole': (200, 0, 255), 'Scratch': (0, 200, 255), 'Water damage': (0, 120, 255), 'Dent': (0, 220, 120)}


class CrateDetector:
    def __init__(self, folder, mode='single', device=None, threshold=None, drop_classes=None, weights_cache=None):
        self.folder = folder; cfg = json.load(open(os.path.join(folder, 'config.json')))
        # classes never reported (default from config.json "drop_classes", e.g. ["Water damage"]); pass [] to report all
        self.drop = set(cfg.get('drop_classes', []) if drop_classes is None else drop_classes)
        if mode not in cfg['modes']: raise ValueError(f"mode must be one of {list(cfg['modes'])}")
        self.mode = mode; self.mcfg = cfg['modes'][mode]
        self.threshold = float(threshold if threshold is not None else self.mcfg['threshold'])
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.models = {}
        for name in self.mcfg['models']:
            spec = cfg['models'][name]; path = _weight_path(folder, spec['file'], weights_cache)
            if spec['kind'] == 'yolo':
                from ultralytics import YOLO
                self.models[name] = (spec, YOLO(path, task='detect'))
            else:
                m = _build_frcnn(spec); m.load_state_dict(torch.load(path, map_location='cpu')); self.models[name] = (spec, m.to(self.dev).eval())

    @staticmethod
    def _load(image):
        if isinstance(image, str): image = Image.open(image)
        if isinstance(image, Image.Image): image = np.asarray(image.convert('RGB'))
        return np.ascontiguousarray(image[:, :, :3])

    @torch.no_grad()
    def _run(self, spec, m, im):
        if spec['kind'] == 'yolo':
            r = m(im[:, :, ::-1].copy(), device=0 if self.dev.type == 'cuda' else 'cpu', imgsz=1024, conf=.001, iou=.7, max_det=100,
                  rect=False, verbose=False)[0]
            B, S, L = r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), (r.boxes.cls.cpu().int() + 1).tolist()
        else:
            x = torch.from_numpy(im.astype(np.float32) / 255.).permute(2, 0, 1).to(self.dev)
            if spec['fp16'] and self.dev.type == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16): o = m([x])[0]
            else: o = m([x])[0]
            B, S, L = o['boxes'].float().tolist(), o['scores'].float().tolist(), o['labels'].tolist()
        r = [{'category_id': int(c), 'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(s)} for (x0, y0, x1, y1), s, c in zip(B, S, L) if x1 > x0 and y1 > y0]
        return sorted(r, key=lambda z: -z['score'])[:100]

    def predict_raw(self, image):
        """All candidate boxes (score >= 0.001 / fused >= 0.01), in 1024x1024 pixel coordinates, before the threshold."""
        im = self._load(image); h, w = im.shape[:2]
        if (h, w) != (1024, 1024): im = np.asarray(Image.fromarray(im).resize((1024, 1024), Image.BILINEAR))
        per = {n: self._run(spec, m, im) for n, (spec, m) in self.models.items()}
        P = per[next(iter(per))] if len(per) == 1 else _wbf(per, len(per))
        return P, (w, h)

    def predict(self, image):
        P, (w, h) = self.predict_raw(image); sx, sy = w / 1024, h / 1024
        defects = [{'class': CLASSES[p['category_id'] - 1], 'score': round(p['score'], 4),
                    'box_xyxy': [round(p['bbox'][0] * sx, 1), round(p['bbox'][1] * sy, 1), round((p['bbox'][0] + p['bbox'][2]) * sx, 1), round((p['bbox'][1] + p['bbox'][3]) * sy, 1)]}
                   for p in P if p['score'] >= self.threshold and CLASSES[p['category_id'] - 1] not in self.drop]
        return {'mode': self.mode, 'threshold': self.threshold, 'crate_flagged': bool(defects), 'image_size': [w, h],
                'damage_areas': merge_areas(defects, w, h), 'defects': defects}

    def report(self, result, frame_id=None):
        """Verdict JSON in the shape of the pipeline diagram (docs/diagrams/Crate-Damage-Production-Pipeline.png):
        verdict, defect_count and one damage record per damaged area. Percentages are of the WHOLE image width/height
        (no crate-face segmentation in this package yet), location = 3x3 grid of the image."""
        W, H = result['image_size']; recs = []
        for k, a in enumerate(result['damage_areas']):
            x0, y0, x1, y1 = a['box_xyxy']; cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            loc = ['upper', 'middle', 'lower'][min(2, int(cy * 3))] + ' ' + ['left', 'centre', 'right'][min(2, int(cx * 3))]
            recs.append({'type': a['class'], 'confidence': a['score'], 'location': loc,
                         'bbox_pct': [round(100 * x0 / W, 1), round(100 * y0 / H, 1), round(100 * (x1 - x0) / W, 1), round(100 * (y1 - y0) / H, 1)],
                         'size_pct': round(100 * max(x1 - x0, y1 - y0) / W, 1), 'box_xyxy_px': a['box_xyxy']})
        return {'frame': frame_id, 'verdict': 'DAMAGE_DETECTED' if recs else 'NO_DEFECT_DETECTED', 'defect_count': len(recs),
                'reason': None, 'mode': result['mode'], 'threshold': result['threshold'], 'damage': recs}

    def draw(self, image, result, out_path=None, use='damage_areas'):
        """Annotated copy of the image: one coloured box + label per damaged area (use='defects' for every raw box)."""
        from PIL import ImageDraw, ImageFont
        im = (Image.open(image) if isinstance(image, str) else (Image.fromarray(image) if isinstance(image, np.ndarray) else image)).convert('RGB').copy()
        d = ImageDraw.Draw(im); fs = max(14, im.width // 60)
        try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', fs)
        except Exception: font = ImageFont.load_default()
        for x in result[use]:
            x0, y0, x1, y1 = x['box_xyxy']; c = COLORS.get(x['class'], (255, 255, 0)); d.rectangle([x0, y0, x1, y1], outline=c, width=max(3, im.width // 300))
            t = f"{x['class']} {x['score']:.2f}"; tw = d.textlength(t, font=font); ty = y0 - fs - 8 if y0 > fs + 8 else y1 + 2
            d.rectangle([x0, ty, x0 + tw + 10, ty + fs + 6], fill=c); d.text((x0 + 5, ty + 2), t, fill=(0, 0, 0), font=font)
        if out_path: im.save(out_path, quality=92)
        return im

    def predict_folder(self, folder, out_jsonl, exts=('.png', '.jpg', '.jpeg', '.bmp')):
        n = 0
        with open(out_jsonl, 'w') as f:
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(exts):
                    r = self.predict(os.path.join(folder, fn)); r['image'] = fn; f.write(json.dumps(r) + '\n'); n += 1
        return n
