"""
Crate defect detector - one file, all models, pick with MODE.

    from crate_detector import CrateDetector
    det = CrateDetector("/Workspace/.../CRATE-DETECTOR-23NOV", mode="single")   # or "ensemble", "bigdefect", "single_bigdefect"
    out = det.predict("crate.png")       # path, PIL image, or RGB numpy array
    out["crate_flagged"]                 # True if any defect >= threshold
    out["defects"]                       # [{"class": "Crack", "score": 0.91, "box_xyxy": [x0, y0, x1, y1]}, ...]

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


class CrateDetector:
    def __init__(self, folder, mode='single', device=None, threshold=None):
        self.folder = folder; cfg = json.load(open(os.path.join(folder, 'config.json')))
        if mode not in cfg['modes']: raise ValueError(f"mode must be one of {list(cfg['modes'])}")
        self.mode = mode; self.mcfg = cfg['modes'][mode]
        self.threshold = float(threshold if threshold is not None else self.mcfg['threshold'])
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.models = {}
        for name in self.mcfg['models']:
            spec = cfg['models'][name]; path = os.path.join(folder, 'models', spec['file'])
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
                   for p in P if p['score'] >= self.threshold]
        return {'mode': self.mode, 'threshold': self.threshold, 'crate_flagged': bool(defects), 'defects': defects}

    def predict_folder(self, folder, out_jsonl, exts=('.png', '.jpg', '.jpeg', '.bmp')):
        n = 0
        with open(out_jsonl, 'w') as f:
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(exts):
                    r = self.predict(os.path.join(folder, fn)); r['image'] = fn; f.write(json.dumps(r) + '\n'); n += 1
        return n
