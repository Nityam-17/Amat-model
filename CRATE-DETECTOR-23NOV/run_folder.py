"""Command line:  python run_folder.py <package_dir> <image_folder> <out.jsonl> [--mode single|ensemble|bigdefect|single_bigdefect] [--threshold T] [--draw out_dir] [--raw-boxes]
--draw saves annotated images with ONE box + label per damaged area (--raw-boxes draws every raw box instead)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crate_detector import CrateDetector
ap = argparse.ArgumentParser(); ap.add_argument('pkg'); ap.add_argument('images'); ap.add_argument('out')
ap.add_argument('--mode', default='single'); ap.add_argument('--threshold', type=float); ap.add_argument('--draw'); ap.add_argument('--raw-boxes', action='store_true')
a = ap.parse_args(); det = CrateDetector(a.pkg, a.mode, threshold=a.threshold)
if a.draw: os.makedirs(a.draw, exist_ok=True)
n = flagged = 0
with open(a.out, 'w') as f:
    for fn in sorted(os.listdir(a.images)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        p = os.path.join(a.images, fn); r = det.predict(p); r['image'] = fn; f.write(json.dumps(r) + '\n'); n += 1; flagged += r['crate_flagged']
        if a.draw:
            det.draw(p, r, os.path.join(a.draw, os.path.splitext(fn)[0] + '.jpg'), use='defects' if a.raw_boxes else 'damage_areas')
print(f'{n} images, {flagged} flagged -> {a.out}')
