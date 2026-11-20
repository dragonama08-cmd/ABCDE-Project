"""نموذج ML بسيط (Decision Tree) لتصنيف درجة الاستعجال من علامات حيوية اصطناعية.

القواعد اللي بتولّد البيانات مستوحاة من مقياس NEWS2 بشكل مبسّط لأغراض تعليمية فقط.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["rr", "spo2", "hr", "sbp", "avpu", "temp"]
FEATURE_AR = {
    "rr": "التنفس",
    "spo2": "الأكسجين",
    "hr": "النبض",
    "sbp": "الضغط",
    "avpu": "الوعي",
    "temp": "الحرارة",
}
LEVELS = ["مستقر", "متوسط", "عاجل"]  # 0, 1, 2


def _label(rr, spo2, hr, sbp, avpu, temp):
    rr_s = np.select([(rr <= 8) | (rr >= 25), rr >= 21, rr <= 11], [3, 2, 1], 0)
    sp_s = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], 0)
    sb_s = np.select([(sbp <= 90) | (sbp >= 220), sbp <= 100, sbp <= 110], [3, 2, 1], 0)
    hr_s = np.select([(hr <= 40) | (hr >= 131), hr >= 111, (hr <= 50) | (hr >= 91)], [3, 2, 1], 0)
    av_s = np.where(avpu > 0, 3, 0)
    tp_s = np.select([temp <= 35.0, temp >= 39.1, (temp <= 36.0) | (temp >= 38.1)], [3, 2, 1], 0)
    comps = np.stack([rr_s, sp_s, sb_s, hr_s, av_s, tp_s])
    total, worst = comps.sum(axis=0), comps.max(axis=0)
    return np.where((total >= 7) | (worst == 3), 2, np.where(total >= 3, 1, 0))


def make_data(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    sev = rng.choice([0, 1, 2], size=n, p=[0.40, 0.30, 0.30])
    rr = np.clip(rng.normal(16 + 5 * sev, 4), 2, 45).round()
    hr = np.clip(rng.normal(78 + 15 * sev, 14), 30, 180).round()
    sbp = np.clip(rng.normal(122 - 12 * sev, 18), 60, 230).round()
    spo2 = np.clip(rng.normal(98 - 2 * sev, 2.2), 70, 100).round()
    p_alt = np.where(sev == 2, 0.5, 0.05)
    avpu = np.where(rng.random(n) < p_alt, rng.integers(1, 4, n), 0)
    temp = np.clip(rng.normal(36.8, 0.6 + 0.4 * sev), 33, 41).round(1)
    X = np.stack([rr, spo2, hr, sbp, avpu, temp], axis=1)
    y = _label(rr, spo2, hr, sbp, avpu, temp)
    flip = rng.random(n) < 0.03  # ضوضاء 3% لتقليد الواقع
    y = np.where(flip, rng.integers(0, 3, n), y)
    return X, y


def train_model():
    X, y = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    counts = {LEVELS[i]: int((y == i).sum()) for i in range(3)}
    return clf, acc, counts


def predict_level(clf, vitals: dict):
    x = np.array([[vitals[f] for f in FEATURES]], dtype=float)
    proba = clf.predict_proba(x)[0]
    full = {LEVELS[int(c)]: float(p) for c, p in zip(clf.classes_, proba)}
    for lv in LEVELS:
        full.setdefault(lv, 0.0)
    best = max(full, key=full.get)
    return best, full
