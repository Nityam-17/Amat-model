# Databricks notebook source
# MAGIC %md
# MAGIC # Crate defect detector
# MAGIC Pick a **MODE** in the widget at the top, then Run all.
# MAGIC * `single` - current model, 1 model, fastest
# MAGIC * `ensemble` - 4 models fused, best on all defect sizes
# MAGIC * `bigdefect` - 4 models fused, best on big defects (>= 32 px)
# MAGIC * `single_bigdefect` - 1 big-defect model, fast
# MAGIC
# MAGIC See GUIDE.txt in this folder for details.

# COMMAND ----------
# CELL 1 - install (only if the cluster does not already have torch/torchvision).
# Use the SAME index URL + trusted host as your other AMAT notebooks. Do NOT paste the URL (it has a token) into shared files.
# %pip install torch torchvision pillow numpy ultralytics==8.4.197 --index-url <YOUR INDEX URL> --trusted-host=serverless-ml-trn.amat.com
# dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.dropdown("MODE", "single", ["single", "ensemble", "bigdefect", "single_bigdefect"])
dbutils.widgets.text("PACKAGE_DIR", "/Workspace/Users/<you>/CRATE-DETECTOR-23NOV")
dbutils.widgets.text("IMAGE_DIR", "/Workspace/Users/<you>/crate-images")
dbutils.widgets.text("THRESHOLD", "")          # empty = the tested threshold for the mode
MODE = dbutils.widgets.get("MODE"); PKG = dbutils.widgets.get("PACKAGE_DIR"); IMG = dbutils.widgets.get("IMAGE_DIR")
THR = dbutils.widgets.get("THRESHOLD").strip() or None

# COMMAND ----------
import sys, os, json
sys.path.append(PKG)
from crate_detector import CrateDetector
det = CrateDetector(PKG, mode=MODE, threshold=THR)
print("mode", det.mode, "| threshold", det.threshold, "| device", det.dev, "| models", list(det.models))

# COMMAND ----------
# one image
files = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
r = det.predict(os.path.join(IMG, files[0])); print(files[0], json.dumps(r, indent=1))

# COMMAND ----------
# whole folder -> table
rows = []
for f in files:
    r = det.predict(os.path.join(IMG, f))
    rows.append({"image": f, "crate_flagged": r["crate_flagged"], "n_defects": len(r["defects"]),
                 "classes": ", ".join(sorted({d["class"] for d in r["defects"]})), "defects_json": json.dumps(r["defects"])})
df = spark.createDataFrame(rows); display(df)
# df.write.mode("overwrite").saveAsTable("<schema>.crate_defects")   # optional
