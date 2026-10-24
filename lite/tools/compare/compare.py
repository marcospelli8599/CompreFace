"""Mismas llamadas que hace el backend de Vero contra CompreFace 1.2.0 y contra compreface-lite, y resumen de diferencias.

Uso: python compare.py <claves.json> [n_pares]
Las fotos son de LFW (conjunto publico de investigacion), en /data/lfw/<Nombre>/<Nombre>_NNNN.jpg.
"""
import io
import json
import os
import random
import statistics as st
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

CF = "http://compreface-api:8080"
LITE = "http://lite:8080"
DATA = Path("/data")
keys = json.load(open(sys.argv[1]))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rng = random.Random(20261024)
FRAC = float(os.environ.get("FRAC", "0.3"))  # parte de las llamadas que se hace con la foto grande (1280 px)

CF_KEY = {"verify": keys["VERIFICATION"], "detect": keys["DETECTION"]}
LITE_KEY = {"verify": "lite-verify-test-key", "detect": "lite-detect-test-key"}


def ensure_lfw():
    if (DATA / "lfw").exists():
        return
    with tarfile.open(DATA / "lfw.tgz") as t:
        t.extractall(DATA)


def jpeg(img: Image.Image, quality=90) -> bytes:
    # PNG (sin perdida) a proposito: cada libreria JPEG decodifica con diferencias de decimas de pixel y el modelo de edad
    # cambia de tramo por eso (ver diagnostico). Con PNG los dos servicios reciben exactamente los mismos pixeles.
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def load(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def call(base, keyset, kind, files, params=None):
    path = "/api/v1/verification/verify" if kind == "verify" else "/api/v1/detection/detect"
    t0 = time.perf_counter()
    try:
        r = requests.post(base + path, headers={"x-api-key": keyset[kind]}, files=files, params=params, timeout=120)
    except requests.RequestException as e:
        return {"status": 0, "error": str(e), "ms": 0.0}
    ms = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except ValueError:
        body = {}
    return {"status": r.status_code, "body": body, "code": body.get("code") if r.status_code != 200 else None, "ms": ms}


def verify_both(a: bytes, b: bytes):
    files = lambda: {"source_image": ("s.png", a, "image/png"), "target_image": ("t.png", b, "image/png")}
    return call(CF, CF_KEY, "verify", files()), call(LITE, LITE_KEY, "verify", files())


def sim(res):
    if res["status"] != 200:
        return None
    return res["body"]["result"][0]["face_matches"][0]["similarity"]


def detect_both(img: bytes, plugins="pose,age"):
    f = lambda: {"file": ("f.png", img, "image/png")}
    p = {"face_plugins": plugins, "limit": 2}
    return call(CF, CF_KEY, "detect", f(), p), call(LITE, LITE_KEY, "detect", f(), p)


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main():
    ensure_lfw()
    people = {}
    for d in sorted((DATA / "lfw").iterdir()):
        files = sorted(d.glob("*.jpg"))
        if files:
            people[d.name] = files
    multi = [p for p, f in people.items() if len(f) >= 2]
    names = list(people)

    # ---------- verificacion ----------
    same = []
    for p in rng.sample(multi, min(N, len(multi))):
        a, b = rng.sample(people[p], 2)
        same.append((a, b))
    diff = []
    while len(diff) < N:
        p, q = rng.sample(names, 2)
        diff.append((rng.choice(people[p]), rng.choice(people[q])))

    rows = []
    for label, pairs in (("misma", same), ("distinta", diff)):
        for a, b in pairs:
            ia, ib = load(a), load(b)
            # tamano LFW (250 px) y tamano de movil (ampliado, que obliga a reducir a 640 como en produccion)
            for variant, (ja, jb) in (
                ("250px", (jpeg(ia), jpeg(ib))),
                ("grande", (jpeg(ia.resize((1280, 1280), Image.LANCZOS)), jpeg(ib.resize((1280, 1280), Image.LANCZOS)))),
            ):
                if variant == "grande" and rng.random() > FRAC:
                    continue
                cf, lt = verify_both(ja, jb)
                rows.append({"tipo": label, "variante": variant, "cf": sim(cf), "lite": sim(lt), "cf_code": cf.get("code"), "lite_code": lt.get("code"),
                             "cf_msg": (cf.get("body") or {}).get("message") if cf["status"] != 200 else None,
                             "cf_ms": cf["ms"], "lite_ms": lt["ms"], "par": f"{a.parent.name}|{b.parent.name}"})

    ok = [r for r in rows if r["cf"] is not None and r["lite"] is not None]
    print(f"\n=== VERIFICACION: {len(rows)} llamadas, {len(ok)} con resultado en ambos ===")
    only_one = [r for r in rows if (r["cf"] is None) != (r["lite"] is None)]
    print(f"error en uno solo: {len(only_one)}   (CompreFace sin cara: {sum(1 for r in rows if r['cf_code']==28)}, lite sin cara: {sum(1 for r in rows if r['lite_code']==28)})")
    for r in only_one[:6]:
        print("   ", r["par"], r["variante"], "cf", r["cf"], r["cf_code"], r.get("cf_msg"), "| lite", r["lite"], r["lite_code"])
    cf = np.array([r["cf"] for r in ok]); lt = np.array([r["lite"] for r in ok])
    d = np.abs(cf - lt)
    print(f"similitud: correlacion {np.corrcoef(cf, lt)[0,1]:.4f}   diferencia media {d.mean():.4f}  p95 {pct(d,95):.4f}  max {d.max():.4f}")
    for tipo in ("misma", "distinta"):
        sel = [r for r in ok if r["tipo"] == tipo]
        print(f"  {tipo:9s} n={len(sel):3d}  CompreFace media {np.mean([r['cf'] for r in sel]):.3f}   lite media {np.mean([r['lite'] for r in sel]):.3f}")
    for name, thr in (("autoVerify 0.85", 0.85), ("samePerson 0.60", 0.60), ("photoMatchReject 0.50", 0.50)):
        flips = [r for r in ok if (r["cf"] >= thr) != (r["lite"] >= thr)]
        print(f"  decisiones distintas con {name}: {len(flips)} de {len(ok)}")
        for r in flips[:5]:
            print(f"      {r['tipo']:8s} {r['variante']:6s} cf {r['cf']:.3f}  lite {r['lite']:.3f}  {r['par']}")
    for tipo, thr in (("misma", 0.85), ("distinta", 0.85)):
        sel = [r for r in ok if r["tipo"] == tipo]
        print(f"  aciertos con 0.85 ({'acepta' if tipo=='misma' else 'rechaza'}): CompreFace {sum((r['cf']>=thr)==(tipo=='misma') for r in sel)}/{len(sel)}   lite {sum((r['lite']>=thr)==(tipo=='misma') for r in sel)}/{len(sel)}")
    print(f"tiempo por llamada: CompreFace mediana {st.median(r['cf_ms'] for r in rows):.0f} ms   lite {st.median(r['lite_ms'] for r in rows):.0f} ms")

    # ---------- deteccion: pose y edad ----------
    singles = [rng.choice(people[p]) for p in rng.sample(names, min(N, len(names)))]
    drows = []
    for path in singles:
        img = load(path)
        for variant, data in (("250px", jpeg(img)), ("grande", jpeg(img.resize((1280, 1280), Image.LANCZOS)))):
            if variant == "grande" and rng.random() > FRAC:
                continue
            cf, lt = detect_both(data)
            def face(res):
                if res["status"] != 200:
                    return None
                faces = res["body"].get("result", [])
                return faces
            fc, fl = face(cf), face(lt)
            row = {"variante": variant, "cf_n": None if fc is None else len(fc), "lite_n": None if fl is None else len(fl),
                   "cf_code": cf.get("code"), "lite_code": lt.get("code"), "cf_ms": cf["ms"], "lite_ms": lt["ms"]}
            if fc and fl and len(fc) == 1 and len(fl) == 1:
                a, b = fc[0], fl[0]
                row.update(cf_yaw=a["pose"]["yaw"], lite_yaw=b["pose"]["yaw"], cf_age=(a["age"]["low"], a["age"]["high"]), lite_age=(b["age"]["low"], b["age"]["high"]),
                           cf_box=a["box"], lite_box=b["box"])
            drows.append(row)

    print(f"\n=== DETECCION (1 cara): {len(drows)} llamadas ===")
    agree_n = sum(1 for r in drows if r["cf_n"] == r["lite_n"] and r["cf_code"] == r["lite_code"])
    print(f"mismo numero de caras / mismo error: {agree_n}/{len(drows)}")
    for r in [r for r in drows if not (r["cf_n"] == r["lite_n"] and r["cf_code"] == r["lite_code"])][:6]:
        print("   distinto:", {k: r[k] for k in ("variante", "cf_n", "lite_n", "cf_code", "lite_code")})
    both = [r for r in drows if "cf_yaw" in r]
    if both:
        dy = np.array([abs(r["cf_yaw"] - r["lite_yaw"]) for r in both])
        print(f"yaw: n={len(both)}  diferencia media {dy.mean():.2f} grados  p95 {pct(dy,95):.2f}  max {dy.max():.2f}")
        sign = sum(1 for r in both if (r["cf_yaw"] >= 0) == (r["lite_yaw"] >= 0))
        print(f"  mismo signo de giro: {sign}/{len(both)}")
        center = sum(1 for r in both if (abs(r["cf_yaw"]) <= 20) == (abs(r["lite_yaw"]) <= 20))
        print(f"  misma decision 'de frente' (|yaw| <= 20): {center}/{len(both)}")
        same_age = sum(1 for r in both if r["cf_age"] == r["lite_age"])
        print(f"edad: mismo tramo {same_age}/{len(both)}")
        diffs = [r for r in both if r["cf_age"] != r["lite_age"]]
        for r in diffs[:6]:
            print(f"   cf {r['cf_age']}  lite {r['lite_age']}  ({r['variante']})")
        minor_cf = {r.get("i"): 0 for r in both}
        for thr in (18,):
            a = sum(1 for r in both if r["cf_age"][1] < thr); b = sum(1 for r in both if r["lite_age"][1] < thr)
            flips = sum(1 for r in both if (r["cf_age"][1] < thr) != (r["lite_age"][1] < thr))
            print(f"  'menor' (edad.high < {thr}): CompreFace {a}  lite {b}  decisiones distintas {flips}")
        iou = []
        for r in both:
            a, b = r["cf_box"], r["lite_box"]
            x1, y1 = max(a["x_min"], b["x_min"]), max(a["y_min"], b["y_min"])
            x2, y2 = min(a["x_max"], b["x_max"]), min(a["y_max"], b["y_max"])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"]) + (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) - inter
            iou.append(inter / ua if ua else 0)
        print(f"caja: solapamiento medio (IoU) {np.mean(iou):.3f}  minimo {min(iou):.3f}")
    print(f"tiempo por llamada: CompreFace mediana {st.median(r['cf_ms'] for r in drows):.0f} ms   lite {st.median(r['lite_ms'] for r in drows):.0f} ms")

    # ---------- casos limite ----------
    print("\n=== CASOS LIMITE ===")
    noface = []
    gen = np.random.default_rng(1)
    noface.append(("ruido", jpeg(Image.fromarray(gen.integers(0, 255, (400, 400, 3), dtype=np.uint8)))))
    noface.append(("blanco", jpeg(Image.new("RGB", (400, 400), "white"))))
    grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
    noface.append(("degradado", jpeg(Image.fromarray(np.stack([grad] * 3, axis=2)))))
    p1, p2 = rng.sample(names, 2)
    i1, i2 = load(people[p1][0]), load(people[p2][0])
    two = Image.new("RGB", (500, 250)); two.paste(i1, (0, 0)); two.paste(i2, (250, 0))
    face = jpeg(i1)
    for nombre, data in noface:
        cf, lt = detect_both(data)
        print(f"detectar {nombre:10s}  CompreFace {cf['status']} cod {cf.get('code')}   lite {lt['status']} cod {lt.get('code')}")
        cfv, ltv = verify_both(data, face)
        print(f"verificar {nombre:9s}+cara CompreFace {cfv['status']} cod {cfv.get('code')}   lite {ltv['status']} cod {ltv.get('code')}")
    cf, lt = detect_both(jpeg(two))
    print(f"detectar dos caras  CompreFace {cf['status']} n={len(cf['body'].get('result', []))}   lite {lt['status']} n={len(lt['body'].get('result', []))}")
    cfv, ltv = verify_both(jpeg(two), face)
    print(f"verificar origen con dos caras  CompreFace {cfv['status']} cod {cfv.get('code')} sim {sim(cfv)}   lite {ltv['status']} cod {ltv.get('code')} sim {sim(ltv)}")
    cfv, ltv = verify_both(face, jpeg(two))
    print(f"verificar destino con dos caras CompreFace {cfv['status']} cod {cfv.get('code')} sim {sim(cfv)}   lite {ltv['status']} cod {ltv.get('code')} sim {sim(ltv)}")
    bad = call(CF, CF_KEY, "detect", {"file": ("f.jpg", b"no soy una imagen", "image/jpeg")}, {"face_plugins": "pose"})
    badl = call(LITE, LITE_KEY, "detect", {"file": ("f.jpg", b"no soy una imagen", "image/jpeg")}, {"face_plugins": "pose"})
    print(f"imagen corrupta     CompreFace {bad['status']} cod {bad.get('code')}   lite {badl['status']} cod {badl.get('code')}")
    wrong = requests.post(CF + "/api/v1/detection/detect", headers={"x-api-key": CF_KEY["verify"]}, files={"file": ("f.jpg", face, "image/jpeg")})
    wrongl = requests.post(LITE + "/api/v1/detection/detect", headers={"x-api-key": LITE_KEY["verify"]}, files={"file": ("f.jpg", face, "image/jpeg")})
    print(f"clave de verificacion en deteccion  CompreFace {wrong.status_code}   lite {wrongl.status_code}")
    json.dump({"verify": rows, "detect": drows}, open("/data/resultados.json", "w"), default=str)


main()


