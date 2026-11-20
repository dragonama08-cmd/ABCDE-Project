import os
import random
import time

import pandas as pd
import streamlit as st

from model import FEATURE_AR, FEATURES, LEVELS, predict_level, train_model
from scenarios import AVPU_TXT, LETTER_NAMES, LETTERS, SCENARIOS, VITAL_LABELS

st.set_page_config(page_title="ABCDE Trainer", page_icon="🚑", layout="centered")
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        direction: rtl; text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return train_model()


clf, model_acc, class_counts = get_model()


# ---------------------------------------------------------------- state
def new_session(sid):
    st.session_state.s = dict(
        sid=sid, start=time.time(), end=None, order=[], actions={},
        current=None, triage=None, call=None, llm=None,
    )


# ---------------------------------------------------------------- timer
def render_timer():
    s = st.session_state.get("s")
    if not s:
        return
    sc = SCENARIOS[s["sid"]]
    end = s["end"] or time.time()
    elapsed = int(end - s["start"])
    limit = sc["time_limit"]
    if elapsed <= limit:
        st.progress(min(1.0, elapsed / limit), text=f"⏱️ {elapsed} ث من {limit} ث")
    elif sc["urgency"] == "عاجل":
        st.error(f"🚨 تأخرت {elapsed - limit} ثانية! في الحالة الحرجة كل ثانية فارقة.")
    else:
        st.warning(f"⏱️ عدّى الوقت المقترح بـ {elapsed - limit} ثانية.")


if hasattr(st, "fragment"):  # تحديث العداد كل ثانية
    render_timer = st.fragment(run_every=1)(render_timer)


# ---------------------------------------------------------------- helpers
def show_vitals(sc, letter):
    keys = sc["steps"][letter]["show"]
    if not keys:
        return
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        label, unit = VITAL_LABELS[k]
        v = sc["vitals"][k]
        col.metric(label, AVPU_TXT[v] if k == "avpu" else f"{v} {unit}".strip())


def compute_score(sc, s):
    order_hits = sum(1 for i, l in enumerate(s["order"]) if l == LETTERS[i])
    order_pts = 20 * order_hits / 5

    mistakes, ratios = [], []
    for l in LETTERS:
        opts = sc["steps"][l]["options"]
        chosen = set(s["actions"].get(l, []))
        right = {t for t, ok in opts if ok}
        wrong = {t for t, ok in opts if not ok}
        good, bad = chosen & right, chosen & wrong
        ratios.append(max(0, len(good) - len(bad)) / len(right))
        for t in right - chosen:
            mistakes.append(f"[{l}] فاتك: {t}")
        for t in bad:
            mistakes.append(f"[{l}] تصرف غلط: {t}")
    action_pts = 50 * sum(ratios) / 5

    diff = abs(LEVELS.index(s["triage"]) - LEVELS.index(sc["urgency"]))
    triage_pts = {0: 20, 1: 8}.get(diff, 0)
    call_pts = 10 if s["call"] == sc["call_ambulance"] else 0

    elapsed = int((s["end"] or time.time()) - s["start"])
    penalty = 10 if (elapsed > sc["time_limit"] and sc["urgency"] == "عاجل") else 0

    total = max(0, min(100, round(order_pts + action_pts + triage_pts + call_pts - penalty)))
    parts = {
        "ترتيب الخطوات (من 20)": round(order_pts),
        "التصرفات (من 50)": round(action_pts),
        "تصنيف الحالة (من 20)": triage_pts,
        "قرار الإسعاف (من 10)": call_pts,
        "خصم التأخير": -penalty,
    }
    return total, parts, mistakes, elapsed


def llm_feedback(sc, s, total, mistakes):
    """تغذية راجعة اختيارية من LLM (محتاجة ANTHROPIC_API_KEY ومكتبة anthropic)."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return "ضيف متغير البيئة ANTHROPIC_API_KEY عشان تفعّل التغذية الراجعة بالذكاء الاصطناعي."
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "إنت مدرّب إسعافات أولية. طالب اتدرب على سيناريو تعليمي (بيانات تجريبية).\n"
            f"السيناريو: {sc['title']} - {sc['intro']}\n"
            f"ترتيب الطالب للخطوات: {' -> '.join(s['order'])} (الصحيح A->B->C->D->E)\n"
            f"تصنيفه للحالة: {s['triage']} (الصحيح: {sc['urgency']})\n"
            f"الدرجة: {total}/100\nالأخطاء:\n" + "\n".join(mistakes or ["مفيش أخطاء"]) +
            "\nاكتب تغذية راجعة مشجعة بالعامية المصرية في 5 أسطر كحد أقصى، "
            "وركّز على أهم نقطتين يحسّنهم. ممنوع تدي تشخيص طبي أو جرعات أدوية."
        )
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:  # noqa: BLE001
        return f"تعذّر الاتصال بالنموذج: {e}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🚑 ABCDE Trainer")
    sid_choice = st.selectbox(
        "اختار السيناريو", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["title"]
    )
    c1, c2 = st.columns(2)
    if c1.button("ابدأ", use_container_width=True, type="primary"):
        new_session(sid_choice)
        st.rerun()
    if c2.button("عشوائي 🎲", use_container_width=True):
        new_session(random.choice(list(SCENARIOS)))
        st.rerun()
    st.divider()
    st.caption(
        "⚠️ المشروع للتعليم والتدريب فقط، وبيستخدم بيانات تجريبية. "
        "مش بديل عن التشخيص الطبي أو الاستشارة المتخصصة. في الطوارئ الحقيقية اتصل 123."
    )

# ---------------------------------------------------------------- main
st.title("نظام تدريب التقييم الأولي للمريض (ABCDE)")
s = st.session_state.get("s")

if not s:
    st.write(
        "اختار سيناريو من القائمة الجانبية واضغط **ابدأ**. هتقيّم المريض بالترتيب "
        "**A → B → C → D → E**، وتختار التصرفات الصح، وفي الآخر تصنّف الحالة وتقرر تطلب الإسعاف ولا لأ."
    )
    st.info("A = مجرى الهواء | B = التنفس | C = الدورة الدموية | D = الوعي | E = الكشف والبيئة")
    st.stop()

sc = SCENARIOS[s["sid"]]
done = s["actions"]
finished = s["triage"] is not None

st.subheader(sc["title"])
st.write(sc["intro"])
render_timer()

# أزرار الخطوات
st.caption("اضغط على الخطوة اللي هتقيّمها دلوقتي (الترتيب بيتحسب في الدرجة):")
cols = st.columns(5)
for col, l in zip(cols, LETTERS):
    if col.button(l, key=f"btn_{l}", disabled=(l in done) or finished,
                  help=LETTER_NAMES[l], use_container_width=True):
        s["current"] = l

# خطوات خلصت
for l in s["order"]:
    with st.expander(f"✅ {l} — {LETTER_NAMES[l]}", expanded=False):
        st.write(sc["steps"][l]["finding"])
        show_vitals(sc, l)
        st.write("**اخترت:** " + ("، ".join(done[l]) if done[l] else "ولا حاجة"))

# الخطوة الحالية
cur = s["current"]
if cur and cur not in done and not finished:
    step = sc["steps"][cur]
    st.markdown(f"### {cur} — {LETTER_NAMES[cur]}")
    st.info(step["finding"])
    show_vitals(sc, cur)
    with st.form(f"form_{s['sid']}_{cur}"):
        picked = st.multiselect("التصرفات المناسبة (اختار كل اللي شايفه صح):",
                                [t for t, _ in step["options"]])
        if st.form_submit_button("تأكيد الخطوة"):
            done[cur] = picked
            s["order"].append(cur)
            s["current"] = None
            st.rerun()

# التصنيف النهائي
if len(done) == 5 and not finished:
    st.markdown("### القرار النهائي")
    with st.form("triage_form"):
        lvl = st.radio("تصنيف الحالة", LEVELS, horizontal=True, index=None)
        call = st.checkbox("🚑 هطلب الإسعاف (123) فورًا")
        if st.form_submit_button("إنهاء وعرض النتيجة"):
            if lvl is None:
                st.warning("اختار تصنيف الحالة الأول.")
            else:
                s["triage"], s["call"], s["end"] = lvl, call, time.time()
                st.rerun()

# النتيجة
if finished:
    total, parts, mistakes, elapsed = compute_score(sc, s)
    st.divider()
    st.header(f"النتيجة: {total} / 100")
    st.progress(total / 100)
    st.table(pd.DataFrame({"النقاط": parts}))
    st.write(f"⏱️ الزمن: {elapsed} ثانية | ترتيبك: {' → '.join(s['order'])}")

    st.subheader("🤖 تصنيف نموذج ML")
    pred, probs = predict_level(clf, sc["vitals"])
    c1, c2, c3 = st.columns(3)
    c1.metric("تصنيف النموذج", pred)
    c2.metric("تصنيفك", s["triage"])
    c3.metric("التصنيف الصحيح", sc["urgency"])
    st.bar_chart(pd.Series(probs, name="احتمال"))
    st.caption(
        f"Decision Tree مدرّب على بيانات اصطناعية ({sum(class_counts.values())} حالة) — "
        f"دقة الاختبار {model_acc:.1%}. النموذج تعليمي وليس أداة تشخيص."
    )
    imp = pd.Series(clf.feature_importances_, index=[FEATURE_AR[f] for f in FEATURES])
    with st.expander("أهمية كل علامة حيوية في قرار النموذج"):
        st.bar_chart(imp)

    st.subheader("📝 مراجعة الأداء")
    if mistakes:
        for m in mistakes:
            st.write("• " + m)
    else:
        st.success("ممتاز! مفيش أخطاء.")
    with st.expander("الشرح العلمي لكل خطوة"):
        for l in LETTERS:
            st.write(f"**{l}:** {sc['steps'][l]['why']}")

    if st.button("🤖 تغذية راجعة من الذكاء الاصطناعي"):
        with st.spinner("بجهّز التغذية الراجعة..."):
            s["llm"] = llm_feedback(sc, s, total, mistakes)
    if s["llm"]:
        st.info(s["llm"])

    if st.button("سيناريو جديد", type="primary"):
        new_session(random.choice(list(SCENARIOS)))
        st.rerun()
