# Databricks notebook source
# MAGIC %md
# MAGIC # Crate defect detector - Databricks
# MAGIC **Only ONE cell is required: "CELL 1 - LOAD".** It rebuilds the weights (stored in git as parts), loads the models
# MAGIC and gives you `det`. After that, use `det` in your own pipeline:
# MAGIC ```python
# MAGIC r = det.predict(image)            # image = file path, PIL image or RGB numpy array
# MAGIC r["damage_areas"]                 # ONE box + label per damaged area (show these)  -> [{class, score, n_boxes, box_xyxy}]
# MAGIC r["defects"]                      # every raw box
# MAGIC det.report(r, frame_id="x.jpg")   # verdict JSON: DAMAGE_DETECTED / NO_DEFECT_DETECTED + damage records
# MAGIC det.draw(image, r)                # annotated PIL image (display() it or .save() it)
# MAGIC ```
# MAGIC Modes: `single` (1 model, fastest) · `ensemble` (4 models, best all sizes, needs ultralytics) ·
# MAGIC `bigdefect` (4 models, best on big defects) · `single_bigdefect` (1 big-defect model). See GUIDE.txt.

# COMMAND ----------
# CELL 0 - install, ONLY if the cluster has no torch / torchvision (ultralytics only for MODE "ensemble").
# Use the SAME index URL + trusted host as your other AMAT notebooks. Do NOT paste that URL into shared files (it has a token).
# %pip install torch torchvision pillow numpy --index-url <YOUR INDEX URL> --trusted-host=serverless-ml-trn.amat.com
# dbutils.library.restartPython()

# COMMAND ----------
# ===================== CELL 1 - LOAD (the only cell you must run) =====================
PKG  = "/Workspace/Users/<you>/Amat-model/CRATE-DETECTOR-23NOV"   # folder that contains crate_detector.py + config.json + models/
MODE = "bigdefect"                                                # "single" | "ensemble" | "bigdefect" | "single_bigdefect"
DROP = []                                                         # e.g. ["Water damage"] to hide that class
import sys; sys.path.append(PKG)
from crate_detector import CrateDetector
det = CrateDetector(PKG, mode=MODE, drop_classes=DROP)            # first run: joins the weight parts into /tmp (~1 min), then loads
print("ready | mode", det.mode, "| threshold", det.threshold, "| device", det.dev, "| models", list(det.models))
# =====================================================================================

# COMMAND ----------
# OPTIONAL - check one image (change the path)
# IMG = "/Workspace/Users/<you>/crate-images/crate_0001.png"
# r = det.predict(IMG); print(det.report(r, frame_id=IMG)); display(det.draw(IMG, r))

# COMMAND ----------
# OPTIONAL - whole folder -> Spark table (+ annotated images)
# import os, json
# IMG_DIR, OUT_DIR = "/Workspace/Users/<you>/crate-images", "/Workspace/Users/<you>/crate-annotated"
# os.makedirs(OUT_DIR, exist_ok=True); rows = []
# for f in sorted(x for x in os.listdir(IMG_DIR) if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))):
#     p = os.path.join(IMG_DIR, f); r = det.predict(p); rep = det.report(r, frame_id=f)
#     det.draw(p, r, os.path.join(OUT_DIR, os.path.splitext(f)[0] + ".jpg"))
#     rows.append({"image": f, "verdict": rep["verdict"], "defect_count": rep["defect_count"],
#                  "types": ", ".join(sorted({d["type"] for d in rep["damage"]})), "report_json": json.dumps(rep)})
# df = spark.createDataFrame(rows); display(df)
# df.write.mode("overwrite").saveAsTable("<schema>.crate_defects")

# COMMAND ----------
# OPTIONAL - self-test (GPU should print ALL OK; on CPU the 4-model modes may differ by a few boxes near the threshold)
# %sh cd /Workspace/Users/<you>/Amat-model/CRATE-DETECTOR-23NOV && python selftest.py
