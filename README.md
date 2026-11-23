# Amat-model - crate defect detector

**Download the full package (code + 8 models, 1.15 GB):** Releases -> `crate-detector-2026-11-23` -> `CRATE-DETECTOR-23NOV.zip`

The `CRATE-DETECTOR-23NOV/` folder here is the same code without the model files (GitHub file size limit).
Unzip the release zip instead to get `models/`. Read `CRATE-DETECTOR-23NOV/GUIDE.txt`.

```python
from crate_detector import CrateDetector
det = CrateDetector("CRATE-DETECTOR-23NOV", mode="single")   # single | ensemble | bigdefect | single_bigdefect
r = det.predict("crate.png")      # r["crate_flagged"], r["defects"] = [{class, score, box_xyxy}]
```

| MODE | models | all defects | big (>=32 px) | ms/img (GPU) |
|---|---|---|---|---|
| single | 1 | 36.8% | 50.8% | 70 |
| ensemble | 4 | 41.7% | 56.7% | 170 |
| bigdefect | 4 | 40.7% | 67.8% | 185 |
| single_bigdefect | 1 | 32.3% | 56.7% | 70 |

**Sample photos to try it:** `samples/CRATE-SAMPLE-IMAGES-10.zip` - 10 crate photos where the models find the defects, with the expected output and drawn boxes (red = model, green = label).

Validation set, threshold so <= 6 of 36 healthy photos get any box.
