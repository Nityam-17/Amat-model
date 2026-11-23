"""Command line:  python run_folder.py <package_dir> <image_folder> <out.jsonl> [--mode single|ensemble|bigdefect|single_bigdefect] [--threshold T] [--draw out_dir]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crate_detector import CrateDetector
ap = argparse.ArgumentParser(); ap.add_argument('pkg'); ap.add_argument('images'); ap.add_argument('out')
ap.add_argument('--mode', default='single'); ap.add_argument('--threshold', type=float); ap.add_argument('--draw')
a = ap.parse_args(); det = CrateDetector(a.pkg, a.mode, threshold=a.threshold)
if a.draw: os.makedirs(a.draw, exist_ok=True)
from PIL import Image, ImageDraw
n = flagged = 0
with open(a.out, 'w') as f:
    for fn in sorted(os.listdir(a.images)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        p = os.path.join(a.images, fn); r = det.predict(p); r['image'] = fn; f.write(json.dumps(r) + '\n'); n += 1; flagged += r['crate_flagged']
        if a.draw:
            im = Image.open(p).convert('RGB'); d = ImageDraw.Draw(im)
            for x in r['defects']: d.rectangle(x['box_xyxy'], outline='red', width=3); d.text((x['box_xyxy'][0] + 3, x['box_xyxy'][1] + 3), f"{x['class']} {x['score']:.2f}", fill='red')
            im.save(os.path.join(a.draw, os.path.splitext(fn)[0] + '.jpg'), quality=90)
print(f'{n} images, {flagged} flagged -> {a.out}')
