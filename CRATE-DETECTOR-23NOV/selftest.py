"""Run after copying the folder:  python selftest.py   (or in Databricks: %run / exec this file)
Checks this machine gives the same answers as ours on 3 sample images, for every mode (or just the modes you name)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from crate_detector import CrateDetector
exp = json.load(open(os.path.join(HERE, 'selftest', 'expected.json'))); ok = True
for mode in (sys.argv[1:] or list(exp)):
    det = CrateDetector(HERE, mode)
    for fn, e in exp[mode].items():
        r = det.predict(os.path.join(HERE, 'selftest', 'images', fn))
        same = r['crate_flagged'] == e['crate_flagged'] and len(r['defects']) == len(e['defects']) and all(
            a['class'] == b['class'] and abs(a['score'] - b['score']) < 0.03 and max(abs(x - y) for x, y in zip(a['box_xyxy'], b['box_xyxy'])) < 4
            for a, b in zip(r['defects'], e['defects']))
        ok &= same; print(f"{mode:17} {fn}: {'OK' if same else 'DIFFERENT'}  flagged={r['crate_flagged']} defects={len(r['defects'])} (expected {len(e['defects'])})")
print('ALL OK' if ok else 'SOME DIFFERENT - small score/box differences near the threshold can happen on another GPU; big differences mean a setup problem')
