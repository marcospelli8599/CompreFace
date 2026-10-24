import json
import numpy as np

d = json.load(open("/data/resultados.json"))
v = [r for r in d["verify"] if r["cf"] is not None and r["lite"] is not None]
v.sort(key=lambda r: -abs(r["cf"] - r["lite"]))
print("--- 6 mayores diferencias de similitud")
for r in v[:6]:
    print(f"  {r['tipo']:8s} {r['variante']:6s} cf {r['cf']:.3f} lite {r['lite']:.3f}  {r['par']}")
diffs = np.array([r["cf"] - r["lite"] for r in v])
print(f"sesgo medio (cf - lite): {diffs.mean():+.4f}   mediana {np.median(diffs):+.4f}")
for tipo in ("misma", "distinta"):
    s = np.array([r["cf"] - r["lite"] for r in v if r["tipo"] == tipo])
    print(f"  {tipo:9s} sesgo {s.mean():+.4f}  |dif| max {np.abs(s).max():.3f}  n={len(s)}")
for variante in ("250px", "grande"):
    s = np.array([abs(r["cf"] - r["lite"]) for r in v if r["variante"] == variante])
    print(f"  {variante:7s} |dif| media {s.mean():.4f}  n={len(s)}")

dd = [r for r in d["detect"] if "cf_yaw" in r]
yaw = np.array([r["cf_yaw"] - r["lite_yaw"] for r in dd])
print(f"\nyaw sesgo (cf - lite): {yaw.mean():+.2f} grados, desviacion {yaw.std():.2f}")
print("yaw: pares cerca del umbral 20 que cambian de decision:")
for r in dd:
    if (abs(r["cf_yaw"]) <= 20) != (abs(r["lite_yaw"]) <= 20):
        print(f"   cf {r['cf_yaw']:+.1f}  lite {r['lite_yaw']:+.1f}  ({r['variante']})")

bins = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
idx = lambda a: bins.index(tuple(a))
ages = [(idx(r["cf_age"]), idx(r["lite_age"]), r) for r in dd]
delta = np.array([c - l for c, l, _ in ages])
print(f"\nedad: tramos de diferencia (cf - lite): " + ", ".join(f"{k:+d}:{int((delta==k).sum())}" for k in sorted(set(delta.tolist()))))
print(f"  sesgo medio {delta.mean():+.2f} tramos")
def iou(a, b):
    x1, y1, x2, y2 = max(a["x_min"], b["x_min"]), max(a["y_min"], b["y_min"]), min(a["x_max"], b["x_max"]), min(a["y_max"], b["y_max"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"]) + (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) - inter
    return inter / ua
print("edad distinta: solapamiento de caja (IoU) y tamano de caja")
for c, l, r in ages:
    if c != l:
        b = r["cf_box"]
        print(f"   cf {bins[c]} lite {bins[l]}  IoU {iou(r['cf_box'], r['lite_box']):.3f}  caja {b['x_max']-b['x_min']}x{b['y_max']-b['y_min']}  {r['variante']}")
low = [(iou(r['cf_box'], r['lite_box'])) for _, _, r in ages]
mism = [i for (c, l, r), i in zip(ages, low) if c != l]
same = [i for (c, l, r), i in zip(ages, low) if c == l]
print(f"IoU medio con edad distinta {np.mean(mism):.3f} / con edad igual {np.mean(same):.3f}")
