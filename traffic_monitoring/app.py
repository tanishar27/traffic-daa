import time
from datetime import datetime

import pandas as pd
import streamlit as st

from vehicle_processing import (
    detect_vehicles, track_vehicles, count_vehicles,
    CONF_THRESHOLD, IMAGE_SIZE, CLASS_NAMES,
)

st.set_page_config(page_title="Traffic Control Room", page_icon="🚦", layout="wide")
st.session_state.setdefault("traffic_report", None)

PAGES = ["Overview", "Vehicle Detection", "Vehicle Tracking", "Traffic Analytics", "Reports"]
DENSITY_TONE = {"Low": "go", "Medium": "slow", "High": "stop"}


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root{--bg:#0e1012;--panel:#15181b;--panel2:#1b1f23;--line:#2a2f35;--line2:#3a4047;
--text:#ece9e2;--muted:#8d9196;--amber:#f5a524;--go:#3ecf8e;--stop:#ef5350;
--mono:'JetBrains Mono',monospace;--head:'Space Grotesk',sans-serif}
html,body,.stApp,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background-color:var(--bg);color:var(--text);
background-image:radial-gradient(ellipse at 50% 0%,transparent 0%,var(--bg) 75%),
linear-gradient(#1c2024 1px,transparent 1px),linear-gradient(90deg,#1c2024 1px,transparent 1px);
background-size:100% 100%,48px 48px,48px 48px;background-attachment:fixed}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.4rem;max-width:1480px;position:relative}
h1,h2,h3,h4{color:var(--text)!important;font-family:var(--head)!important;letter-spacing:-.02em}
p,.stMarkdown,.stCaption{color:var(--muted)}

/* panels */
[class*="st-key-panel"]{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.ptitle{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.ptitle .lbl b{color:var(--text);font-weight:600}

/* sidebar */
[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarContent"]{padding-top:8px}
.brand{display:flex;align-items:center;gap:12px;padding:4px 0 18px;border-bottom:1px solid var(--line);margin-bottom:16px}
.signal{display:flex;flex-direction:column;gap:3px;padding:5px;background:#090a0b;border:1px solid var(--line2);border-radius:7px}
.signal i{width:9px;height:9px;border-radius:50%;background:#2a2f35}
.signal i.r{background:var(--stop);box-shadow:0 0 8px var(--stop)}
.signal i.a{background:var(--amber);box-shadow:0 0 8px var(--amber)}
.signal i.g{background:var(--go);box-shadow:0 0 8px var(--go)}
.brand b{font-family:var(--head);font-size:17px;color:var(--text);display:block;line-height:1.1}
.brand span{font-family:var(--mono);font-size:10.5px;color:var(--muted);letter-spacing:.14em}

[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;counter-reset:nav}
[data-testid="stSidebar"] [role="radiogroup"]>label>div:first-child{display:none}
[data-testid="stSidebar"] [role="radiogroup"]>label{width:100%;padding:10px 12px;border-radius:6px;margin:0!important;border-left:2px solid transparent;transition:background .15s}
[data-testid="stSidebar"] [role="radiogroup"]>label p{color:var(--muted)!important;font-weight:500;font-size:14.5px}
[data-testid="stSidebar"] [role="radiogroup"]>label p::before{counter-increment:nav;content:"0" counter(nav);font-family:var(--mono);font-size:11px;
color:#5a6068;margin-right:12px}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover{background:var(--panel2)}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked){background:var(--panel2);border-left-color:var(--amber)}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked) p{color:var(--text)!important}
[data-testid="stSidebar"] [role="radiogroup"]>label:has(input:checked) p::before{color:var(--amber)}

.kv{border-top:1px solid var(--line);margin-top:18px;padding-top:14px}
.kv div{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px dashed var(--line);font-size:13px}
.kv div span{color:var(--muted)}.kv div b{font-family:var(--mono);color:var(--text);font-weight:600}

/* top bar */
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
padding-bottom:14px;margin-bottom:24px;border-bottom:1px solid var(--line)}
.crumb{font-family:var(--mono);font-size:12px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}
.crumb b{color:var(--amber);font-weight:600}
.status{display:flex;align-items:center;gap:18px;font-family:var(--mono);font-size:12px;color:var(--muted)}
.status .clock{color:var(--text);font-size:15px;font-weight:600}
.led{display:inline-flex;align-items:center;gap:8px;padding:5px 10px;border:1px solid var(--line2);border-radius:999px;
letter-spacing:.1em;color:var(--go)}
.led::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--go);animation:blink 2s infinite}
@keyframes blink{50%{opacity:.25}}

/* page head */
.head{margin-bottom:20px}.head h2{margin:6px 0 4px!important;font-size:30px!important;font-weight:700!important}
.head p{margin:0;font-size:15px}

/* hero */
.hero{display:grid;grid-template-columns:1fr 1fr;gap:36px;align-items:center;margin-bottom:28px}
.hero h1{font-size:54px!important;font-weight:700!important;line-height:1.02;margin:12px 0 16px!important}
.hero h1 em{font-style:normal;color:var(--amber)}
.hero p{font-size:16px;line-height:1.65;max-width:500px;margin:0 0 22px}
.classes{display:flex;gap:8px;flex-wrap:wrap}
.classes span{font-family:var(--mono);font-size:12px;padding:6px 10px;border:1px solid var(--line2);border-radius:4px;color:var(--text)}
.cam{position:relative;border:1px solid var(--line2);border-radius:10px;overflow:hidden;background:#0a0c0e}
.cam svg{display:block;width:100%}
.cam .bar{position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;padding:10px 12px;
font-family:var(--mono);font-size:11px;letter-spacing:.1em;color:var(--text);background:linear-gradient(#0a0c0ecc,transparent)}
.rec{color:var(--stop)}.rec::before{content:"● "}
@media(max-width:900px){.hero{grid-template-columns:1fr}.hero h1{font-size:40px!important}}

/* modules */
.mod .num{font-family:var(--mono);font-size:12px;color:var(--amber);letter-spacing:.1em}
.mod h4{margin:10px 0 6px!important;font-size:19px!important;font-weight:600!important}
.mod p{margin:0 0 16px;font-size:13.5px;line-height:1.55;min-height:62px}
[class*="st-key-panel_mod"]{transition:border-color .2s}
[class*="st-key-panel_mod"]:hover{border-color:var(--amber)}

/* buttons */
.stButton>button{background:var(--amber);border:1px solid var(--amber);border-radius:6px;min-height:44px;padding:8px 22px;
font-family:var(--mono);font-weight:700;letter-spacing:.06em;text-transform:uppercase;font-size:13px;transition:filter .15s}
.stButton>button,.stButton>button *{color:#17130a!important}
.stButton>button:hover{filter:brightness(1.1);border-color:var(--amber)}
[class*="st-key-panel_mod"] .stButton>button{background:transparent;border-color:var(--line2)}
[class*="st-key-panel_mod"] .stButton>button,[class*="st-key-panel_mod"] .stButton>button *{color:var(--text)!important}
[class*="st-key-panel_mod"] .stButton>button:hover{border-color:var(--amber);background:var(--panel2)}
.stDownloadButton>button{background:transparent;border:1px solid var(--line2);border-radius:6px;min-height:44px;
font-family:var(--mono);font-weight:600;font-size:13px}
.stDownloadButton>button,.stDownloadButton>button *{color:var(--text)!important}
.stDownloadButton>button:hover{border-color:var(--amber)}

/* readouts */
.ro{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}
.ro .v{font-family:var(--mono);font-size:38px;font-weight:700;color:var(--text);line-height:1.1;margin-top:8px;
display:flex;align-items:center;gap:12px}
.ro.go .v{color:var(--go)}.ro.slow .v{color:var(--amber)}.ro.stop .v{color:var(--stop)}
.ro .signal{flex-direction:row;padding:4px 5px}

/* feed */
[data-testid="stImage"] img{border-radius:6px;display:block}
.feed{min-height:440px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
border-radius:6px;background:repeating-linear-gradient(135deg,#0b0d0f 0 12px,#0e1114 12px 24px);
border:1px solid var(--line);color:var(--muted);font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
.feed b{font-family:var(--head);font-size:20px;letter-spacing:0;text-transform:none;color:var(--text);font-weight:600}

/* misc */
[data-testid="stAlert"]{background:transparent!important;border:none!important}
[data-testid="stAlert"]>div,[data-testid="stAlertContainer"]{background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:8px!important}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{color:var(--text)!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:6px;overflow:hidden}
.foot{margin-top:36px;padding-top:14px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;
letter-spacing:.1em;color:var(--muted);text-transform:uppercase}
</style>
""", unsafe_allow_html=True)


def html(markup, target=st):
    target.markdown("\n".join(l.strip() for l in markup.splitlines() if l.strip()), unsafe_allow_html=True)


def go(p):
    st.session_state.nav = p


def signal(state):
    lit = {"stop": "r", "slow": "a", "go": "g"}.get(state)
    return '<span class="signal">' + "".join(
        f'<i class="{c if c == lit else ""}"></i>' for c in ("r", "a", "g")) + "</span>"


def readout(slot, label, value, tone=""):
    light = signal(tone) if tone else ""
    html(f'<div class="ro {tone}"><div class="lbl">{label}</div><div class="v">{light}{value}</div></div>', slot)


def ptitle(left, right=""):
    html(f'<div class="ptitle"><div class="lbl"><b>{left}</b></div><div class="lbl">{right}</div></div>')


def head(step, title, sub):
    html(f'<div class="head"><div class="lbl">{step}</div><h2>{title}</h2><p>{sub}</p></div>')


def feed(text):
    return f'<div class="feed">No signal<b>{text}</b></div>'


def cam_art():
    boxes = [(188, 96, 30, 20, "CAR 0.91"), (246, 134, 58, 44, "TRUCK 0.87"),
             (116, 182, 30, 50, "MOTO 0.74"), (300, 190, 66, 50, "CAR 0.95")]
    body = ""
    for x, y, w, h, lab in boxes:
        body += (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#262b31"/>'
                 f'<rect x="{x + w * .15}" y="{y + h * .15}" width="{w * .7}" height="{h * .25}" rx="2" fill="#4a525b"/>'
                 f'<rect x="{x - 4}" y="{y - 4}" width="{w + 8}" height="{h + 8}" fill="none" stroke="#f5a524" stroke-width="1.5"/>'
                 f'<rect x="{x - 4}" y="{y - 16}" width="{len(lab) * 6 + 8}" height="12" fill="#f5a524"/>'
                 f'<text x="{x}" y="{y - 7}" fill="#17130a" font-size="8.5" font-family="monospace" font-weight="700">{lab}</text>')
    return f"""<div class="cam"><div class="bar"><span>CAM 01 · NORTH APPROACH</span><span class="rec">REC</span></div>
      <svg viewBox="0 0 420 270">
      <rect width="420" height="270" fill="#0d0f11"/>
      <polygon points="186,40 234,40 420,270 0,270" fill="#14171a"/>
      <line x1="210" y1="40" x2="210" y2="270" stroke="#ece9e2" stroke-opacity=".3" stroke-dasharray="10 10"/>
      <line x1="198" y1="40" x2="90" y2="270" stroke="#ece9e2" stroke-opacity=".2" stroke-dasharray="10 10"/>
      <line x1="222" y1="40" x2="330" y2="270" stroke="#ece9e2" stroke-opacity=".2" stroke-dasharray="10 10"/>
      <line x1="0" y1="168" x2="420" y2="168" stroke="#3ecf8e" stroke-width="1.5" stroke-dasharray="4 4"/>
      <text x="8" y="162" fill="#3ecf8e" font-size="8.5" font-family="monospace">COUNTING LINE</text>
      {body}</svg></div>"""


def settings_panel(key):
    with st.container(key=f"panel_{key}_settings"):
        ptitle("Model settings")
        html('<div class="kv" style="border:0;margin:0;padding:0">' + "".join(
            f"<div><span>{k}</span><b>{v}</b></div>" for k, v in
            [("Confidence", f"{CONF_THRESHOLD:.2f}"), ("Input size", f"{IMAGE_SIZE}px"),
             ("Classes", len(CLASS_NAMES)), ("Tracker", "ByteTrack")]) + "</div>")


def result_cards(cats, report, key):
    a, b = st.columns(2, gap="medium")
    with a, st.container(key=f"panel_{key}_chart"):
        ptitle("Vehicles by type", "line crossings")
        st.bar_chart(cats.set_index("Vehicle Type"), color="#f5a524")
    with b, st.container(key=f"panel_{key}_table"):
        ptitle("Traffic report")
        st.dataframe(report, width="stretch", hide_index=True)
    st.download_button("Download CSV report", report.to_csv(index=False).encode("utf-8"),
                       "traffic_report.csv", "text/csv", key=f"{key}_download")


def summary_row(saved):
    c = st.columns(3, gap="medium")
    readout(c[0], "Total vehicles crossed", saved["total"])
    readout(c[1], "Currently visible", saved["visible"])
    readout(c[2], "Traffic density", saved["density"], DENSITY_TONE.get(saved["density"], ""))


def live_page(key, step, title, sub, source, label):
    head(step, title, sub)
    start = st.button(f"Start {title.split()[-1].lower()}", key=key)
    left, right = st.columns([3, 1.1], gap="medium")
    with left, st.container(key=f"panel_{key}_feed"):
        ptitle(f"Live feed · {title}", "CAM 01")
        video = st.empty()
        video.markdown(feed("Press start to run the model"), unsafe_allow_html=True)
    with right:
        metric = st.empty()
        readout(metric, label, 0)
        settings_panel(key)
    if start:
        try:
            with st.spinner(f"Running {title.lower()}..."):
                for frame, count in source():
                    video.image(frame, channels="BGR", width="stretch")
                    readout(metric, label, count)
                    time.sleep(0.02)
            st.success(f"{title} completed.")
        except Exception as e:
            st.error(f"{title} failed: {e}")


with st.sidebar:
    html(f'<div class="brand">{signal("go")}<div><b>Traffic Control</b><span>YOLO · BYTETRACK</span></div></div>')
    page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="nav")
    html('<div class="kv">' + "".join(
        f"<div><span>{k}</span><b>{v}</b></div>" for k, v in
        [("Confidence", f"{CONF_THRESHOLD:.2f}"), ("Input size", f"{IMAGE_SIZE}px"),
         ("Classes", len(CLASS_NAMES)), ("Tracker", "ByteTrack")]) + "</div>")

now = datetime.now()
html(f"""
<div class="top">
  <div class="crumb">Control room / <b>{page}</b></div>
  <div class="status"><span>{now:%a %d %b %Y}</span><span class="clock">{now:%H:%M}</span><span class="led">ENGINE READY</span></div>
</div>""")


if page == "Overview":
    classes = "".join(f"<span>{n.upper()}</span>" for n in CLASS_NAMES.values())
    html(f"""
    <div class="hero">
      <div>
        <div class="lbl">Smart-city traffic monitoring</div>
        <h1>Every vehicle,<br><em>counted</em> and tracked.</h1>
        <p>Run YOLO on traffic video to detect cars, motorcycles, buses and trucks, follow each one with
        ByteTrack, count line crossings and rate how congested the road is.</p>
        <div class="classes">{classes}</div>
      </div>
      {cam_art()}
    </div>""")

    modules = [
        ("Vehicle Detection", "01", "Detection", "Find and label every vehicle in each frame of the video."),
        ("Vehicle Tracking", "02", "Tracking", "Give each vehicle a unique ID and follow it across frames."),
        ("Traffic Analytics", "03", "Analytics", "Count vehicles crossing the line and rate the traffic density."),
        ("Reports", "04", "Reports", "Review the latest results and download them as a CSV file."),
    ]
    for col, (target, num, title, text) in zip(st.columns(4, gap="medium"), modules):
        with col, st.container(key=f"panel_mod_{title.lower()}"):
            html(f'<div class="mod"><div class="num">{num}</div><h4>{title}</h4><p>{text}</p></div>')
            st.button("Open →", key=f"open_{title}", on_click=go, args=(target,), width="stretch")

    st.write("")
    saved = st.session_state.traffic_report
    if saved is None:
        st.info("No report yet. Run Traffic Analytics to generate your first traffic report.")
    else:
        html('<div class="lbl" style="margin-bottom:10px">Latest traffic report</div>')
        summary_row(saved)

elif page == "Vehicle Detection":
    live_page("detect", "Module 01", "Vehicle Detection",
              "Detect and classify cars, motorcycles, buses and trucks using YOLO.",
              detect_vehicles, "Vehicles detected")

elif page == "Vehicle Tracking":
    live_page("tracking", "Module 02", "Vehicle Tracking",
              "Follow every vehicle with a unique tracking ID using ByteTrack.",
              track_vehicles, "Currently tracked")

elif page == "Traffic Analytics":
    head("Module 03", "Traffic Analytics", "Count vehicles crossing the line and estimate traffic density.")
    start = st.button("Start analysis", key="analytics")

    left, right = st.columns([3, 1.1], gap="medium")
    with left, st.container(key="panel_analytics_feed"):
        ptitle("Live feed · Traffic Analytics", "CAM 01")
        video = st.empty()
        video.markdown(feed("Press start to run traffic analysis"), unsafe_allow_html=True)
    with right:
        slots = {k: st.empty() for k in ("total", "visible", "density")}
        labels = {"total": "Total vehicles crossed", "visible": "Currently visible", "density": "Traffic density"}

        def show(stats):
            for k, s in slots.items():
                v = stats.get(k, "—")
                readout(s, labels[k], v, DENSITY_TONE.get(v, "") if k == "density" else "")

        show({"total": 0, "visible": 0})

    if start:
        stats = None
        try:
            with st.spinner("Analysing traffic video..."):
                for frame, stats in count_vehicles():
                    video.image(frame, channels="BGR", width="stretch")
                    show(stats)
                    time.sleep(0.02)

            if stats is None:
                st.warning("No video frames were processed.")
            else:
                st.success("Traffic analysis completed.")
                cats = pd.DataFrame({
                    "Vehicle Type": ["Cars", "Motorcycles", "Buses", "Trucks"],
                    "Vehicles Counted": [stats["Car"], stats["Motorcycle"], stats["Bus"], stats["Truck"]],
                })
                report = pd.DataFrame({
                    "Metric": ["Total Vehicles Crossed", "Currently Visible", "Traffic Density",
                               "Cars", "Motorcycles", "Buses", "Trucks"],
                    "Value": [str(v) for v in (stats["total"], stats["visible"], stats["density"],
                                               stats["Car"], stats["Motorcycle"], stats["Bus"], stats["Truck"])],
                })
                st.session_state.traffic_report = {
                    "total": stats["total"], "visible": stats["visible"], "density": stats["density"],
                    "categories": cats, "report_data": report,
                }
                for col, (name, n) in zip(st.columns(4, gap="medium"), cats.values):
                    readout(col, name, n)
                result_cards(cats, report, "analytics")
        except Exception as e:
            st.error(f"Traffic analysis failed: {e}")

elif page == "Reports":
    head("Module 04", "Traffic Reports", "View and download your most recent traffic report.")
    saved = st.session_state.traffic_report

    if saved is None:
        st.info("No report available yet. Run Traffic Analytics first.")
    else:
        summary_row(saved)
        st.write("")
        result_cards(saved["categories"], saved["report_data"], "report")

html('<div class="foot">Traffic Control Room · Python · YOLO · OpenCV · ByteTrack · Streamlit</div>')
