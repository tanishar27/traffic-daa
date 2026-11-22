import time
from datetime import datetime

import pandas as pd
import streamlit as st

from vehicle_processing import (
    detect_vehicles, track_vehicles, count_vehicles,
    CONF_THRESHOLD, IMAGE_SIZE, CLASS_NAMES,
)

st.set_page_config(page_title="AI Traffic Monitoring System", page_icon="🚦", layout="wide")
st.session_state.setdefault("traffic_report", None)

PAGES = {"Overview": "◫", "Vehicle Detection": "◉", "Vehicle Tracking": "⌖",
         "Traffic Analytics": "▥", "Reports": "▤"}
CLASS_ICONS = {"Car": "🚗", "Motorcycle": "🏍️", "Bus": "🚌", "Truck": "🚚"}


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
:root{--card:rgba(255,255,255,.065);--hover:rgba(255,255,255,.10);--line:rgba(255,255,255,.11);
--text:#f1f5ff;--muted:#93a0c0;--green:#34d399;--cyan:#38bdf8;--violet:#8b7cff}
html,body,[class*="css"],.stApp{font-family:'Plus Jakarta Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 15% 10%,rgba(56,120,255,.34),transparent 45%),
radial-gradient(ellipse at 90% 20%,rgba(139,124,255,.22),transparent 45%),
radial-gradient(ellipse at 60% 100%,rgba(20,184,166,.16),transparent 50%),
linear-gradient(180deg,#0b1330,#0a1026);background-attachment:fixed;color:var(--text)}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1560px}
h1,h2,h3,h4{color:var(--text)!important}
p,.stMarkdown,.stCaption{color:var(--muted)}

[class*="st-key-card"],[data-testid="stMetric"],.b-card{background:var(--card);backdrop-filter:blur(20px);
border:1px solid var(--line);border-radius:26px;box-shadow:0 14px 44px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.1)}
[class*="st-key-card"]{padding:22px 24px}
[data-testid="stMetric"]{padding:20px 24px;transition:.25s}
[data-testid="stMetric"]:hover,.b-card:hover{background:var(--hover);transform:translateY(-3px)}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-weight:600}
[data-testid="stMetricValue"]{color:#fff!important;font-weight:800}
[data-testid="stSidebar"]{background:rgba(255,255,255,.04);backdrop-filter:blur(26px);border-right:1px solid var(--line)}

[data-testid="stRadioOption"]>div>div:first-child{display:none}
[data-testid="stSidebar"] [role="radiogroup"]{gap:8px}
[data-testid="stRadioOption"]{padding:12px 16px;border-radius:16px;border:1px solid transparent;margin:0!important;transition:.25s}
[data-testid="stRadioOption"] p{color:var(--muted)!important;font-weight:600;font-size:15px}
[data-testid="stRadioOption"]:hover{background:var(--card);border-color:var(--line)}
[data-testid="stRadioOption"]:has(input:checked){background:linear-gradient(135deg,rgba(56,189,248,.26),rgba(139,124,255,.26));
border-color:rgba(56,189,248,.5);box-shadow:0 8px 26px rgba(56,189,248,.22)}
[data-testid="stRadioOption"]:has(input:checked) p{color:#fff!important}

.stButton>button{background:linear-gradient(135deg,#22c7a5,#38bdf8 55%,#8b7cff);border:none;border-radius:16px;
min-height:50px;padding:10px 28px;font-weight:800;box-shadow:0 10px 30px rgba(56,189,248,.35);transition:.25s}
.stButton>button,.stButton>button *{color:#07102a!important}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 14px 38px rgba(52,211,153,.5)}
.stDownloadButton>button{background:var(--card);color:var(--cyan);border:1px solid rgba(56,189,248,.5);border-radius:16px;min-height:48px;font-weight:700}
[data-testid="stAlert"]{background:transparent!important;border:none!important}
[data-testid="stAlert"]>div,[data-testid="stAlertContainer"]{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:18px!important}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{color:var(--text)!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:18px;overflow:hidden}

[data-testid="stImage"]{position:relative}
[data-testid="stImage"] img{border-radius:18px;display:block}
[data-testid="stImage"]::before,[data-testid="stImage"]::after{content:"";position:absolute;width:38px;height:38px;
border:3px solid var(--green);pointer-events:none;z-index:3;filter:drop-shadow(0 0 6px rgba(52,211,153,.8))}
[data-testid="stImage"]::before{top:10px;left:10px;border-right:0;border-bottom:0;border-radius:14px 0 0 0}
[data-testid="stImage"]::after{bottom:10px;right:10px;border-left:0;border-top:0;border-radius:0 0 14px 0}

.topbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:18px;margin-bottom:22px}
.time{font-size:40px;font-weight:800;color:#fff;line-height:1}.time small{font-size:17px;color:var(--muted);margin-left:6px}
.date{color:var(--muted);font-size:14px;margin-top:6px}
.pill{flex:1;max-width:460px;min-width:240px;display:flex;align-items:center;gap:14px;padding:14px 20px;border-radius:999px;
background:var(--card);border:1px solid var(--line);color:var(--muted);font-size:14.5px}.pill b{color:#fff}
.chip{padding:11px 18px;border-radius:999px;background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.45);
color:#6ee7b7;font-weight:700;font-size:13px;display:flex;align-items:center;gap:10px}
.avatar{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;color:#07102a;
background:linear-gradient(135deg,#22c7a5,#8b7cff);box-shadow:0 0 22px rgba(139,124,255,.5)}
.dot{width:10px;height:10px;border-radius:50%;background:var(--green);animation:pulse 1.8s infinite;flex:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.7)}70%,100%{box-shadow:0 0 0 11px rgba(52,211,153,0)}}
.head h2{margin:0 0 4px!important;font-size:28px!important;font-weight:800!important}.head p{margin:0 0 18px}
.hero{display:grid;grid-template-columns:1.05fr 1fr;gap:30px;align-items:center;padding:34px;margin-bottom:22px;
background:linear-gradient(120deg,rgba(34,199,165,.14),rgba(139,124,255,.16));border-radius:30px}
.hero h1{font-size:44px!important;font-weight:800!important;line-height:1.1;margin:14px 0 12px!important;
background:linear-gradient(90deg,#fff,#7dd3fc 60%,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{font-size:16px;line-height:1.7;margin:0 0 20px;max-width:520px}
.tag{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;background:rgba(255,255,255,.08);
border:1px solid var(--line);color:#7dd3fc;font-size:12px;font-weight:700;letter-spacing:2px}
.types{display:flex;flex-wrap:wrap;gap:10px}.types span{padding:9px 16px;border-radius:999px;background:rgba(255,255,255,.08);
border:1px solid var(--line);color:var(--text);font-size:13.5px;font-weight:600}
@media(max-width:900px){.hero{grid-template-columns:1fr}}
.mod .ico{width:56px;height:56px;border-radius:18px;font-size:26px;margin-bottom:14px}
.mod h4{margin:0 0 6px!important;font-size:19px!important;font-weight:800!important}
.mod p{margin:0 0 12px;font-size:13.5px;line-height:1.6;min-height:66px}
[class*="st-key-card_mod"]{transition:.3s}[class*="st-key-card_mod"]:hover{transform:translateY(-6px);border-color:rgba(56,189,248,.6)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
[class*="st-key-card"] .grid2.tiles{grid-template-columns:1fr}
.tile{display:flex;align-items:center;gap:12px;padding:14px;border-radius:18px;background:rgba(255,255,255,.06);
border:1px solid var(--line);min-width:0}
.tile b{display:block;color:#fff;font-size:17px;font-weight:800;line-height:1.15}.tile span{color:var(--muted);font-size:11.5px;font-weight:600}
.ico{width:38px;height:38px;border-radius:12px;flex:none;display:flex;align-items:center;justify-content:center;font-weight:800;color:#07102a}
.o{background:linear-gradient(135deg,#fdba74,#fb923c)}.v{background:linear-gradient(135deg,#c4b5fd,#8b7cff)}
.gr{background:linear-gradient(135deg,#6ee7b7,#34d399)}.cy{background:linear-gradient(135deg,#7dd3fc,#38bdf8)}
.feed{min-height:420px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;border:2px dashed rgba(255,255,255,.16);
border-radius:20px;color:var(--muted);font-weight:600;background:rgba(0,0,0,.18)}.feed div{font-size:46px}
.ctitle{color:#fff;font-weight:700;font-size:16px;margin-bottom:14px}
.logo{display:flex;align-items:center;gap:12px}.logo i{width:46px;height:46px;border-radius:15px;display:flex;align-items:center;justify-content:center;
font-style:normal;font-size:22px;background:linear-gradient(135deg,#22c7a5,#38bdf8 55%,#8b7cff);box-shadow:0 0 24px rgba(56,189,248,.5)}
.logo b{color:#fff;font-size:17px;font-weight:800;display:block}.logo span{color:var(--muted);font-size:11px;letter-spacing:1.6px}
</style>
""", unsafe_allow_html=True)


def html(markup):
    st.markdown("\n".join(l.strip() for l in markup.splitlines() if l.strip()), unsafe_allow_html=True)


def tiles(items, extra=""):
    body = "".join(
        f'<div class="tile"><div class="ico {c}">{i}</div><div><b>{b}</b><span>{s}</span></div></div>'
        for c, i, b, s in items
    )
    return f'<div class="grid2 {extra}">{body}</div>'


def card(area, title, sub, body):
    return f'<div class="b-card {area}"><div class="b-title">{title}</div><div class="b-sub">{sub}</div>{body}</div>'


def go(p):
    st.session_state.nav = p


def bracket(x, y, w, h, label, c=34):
    k = 7
    d = (f"M{x},{y+k}V{y}H{x+k} M{x+w-k},{y}H{x+w}V{y+k} "
         f"M{x+w},{y+h-k}V{y+h}H{x+w-k} M{x+k},{y+h}H{x}V{y+h-k}")
    return (f'<path d="{d}" stroke="#34d399" stroke-width="2" fill="none" stroke-linecap="round"/>'
            f'<text x="{x}" y="{y-5}" fill="#34d399" font-size="9" font-weight="700">{label}</text>')


def hero_art():
    cars = [(187, 82, 26, 17, "Car"), (236, 122, 54, 42, "Truck"), (112, 172, 34, 58, "Motorcycle"),
            (298, 178, 64, 48, "Car")]
    body = "".join(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#2a3f80" stroke="#5b7be8" stroke-width="1.2"/>'
        f'<rect x="{x+w*.15}" y="{y+h*.15}" width="{w*.7}" height="{h*.28}" rx="3" fill="#8fb2ff" opacity=".55"/>'
        + bracket(x - 4, y - 4, w + 8, h + 8, lab)
        for x, y, w, h, lab in cars
    )
    return f"""<svg viewBox="0 0 420 260" style="width:100%;border-radius:22px">
      <defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#16265a"/><stop offset="1" stop-color="#0a1230"/></linearGradient></defs>
      <rect width="420" height="260" fill="url(#bg)"/>
      <polygon points="190,30 230,30 420,260 0,260" fill="#0d1636" stroke="#2a3f80"/>
      <line x1="210" y1="30" x2="210" y2="260" stroke="#fff" stroke-opacity=".35" stroke-dasharray="10 10"/>
      <line x1="200" y1="30" x2="95" y2="260" stroke="#fff" stroke-opacity=".25" stroke-dasharray="10 10"/>
      <line x1="220" y1="30" x2="325" y2="260" stroke="#fff" stroke-opacity=".25" stroke-dasharray="10 10"/>
      {body}</svg>"""


def head(title, sub):
    html(f'<div class="head"><h2>{title}</h2><p>{sub}</p></div>')


def feed(text):
    return f'<div class="feed"><div>🎥</div>{text}</div>'


SETTINGS = [("o", "◎", f"{CONF_THRESHOLD:.2f}", "Confidence"), ("v", "▣", f"{IMAGE_SIZE}px", "Input size"),
            ("gr", "⬡", len(CLASS_NAMES), "Classes"), ("cy", "⇄", "ByteTrack", "Tracker")]


def result_cards(cats, report, key):
    a, b = st.columns(2, gap="medium")
    with a, st.container(key=f"card_{key}_chart"):
        html('<div class="ctitle">Vehicle Count Chart</div>')
        st.bar_chart(cats.set_index("Vehicle Type"), color="#38bdf8")
    with b, st.container(key=f"card_{key}_table"):
        html('<div class="ctitle">Traffic Report</div>')
        st.dataframe(report, width="stretch", hide_index=True)
    st.download_button("⬇  Download CSV Report", report.to_csv(index=False).encode("utf-8"),
                       "traffic_report.csv", "text/csv", key=f"{key}_download")


def live_page(key, title, sub, source, label):
    head(title, sub)
    start = st.button(f"▶  Start {title}", key=key)
    left, right = st.columns([3, 1.15], gap="medium")
    with left, st.container(key=f"card_{key}_feed"):
        html(f'<div class="ctitle">● Live Feed · {title}</div>')
        video = st.empty()
        video.markdown(feed(f"Press Start to begin"), unsafe_allow_html=True)
    with right:
        metric = st.empty()
        metric.metric(label, 0)
        with st.container(key=f"card_{key}_info"):
            html('<div class="ctitle">Settings</div>' + tiles(SETTINGS, "tiles"))
    if start:
        try:
            with st.spinner(f"Running {title.lower()}..."):
                for frame, count in source():
                    video.image(frame, channels="BGR", width="stretch")
                    metric.metric(label, count)
                    time.sleep(0.02)
            st.success(f"{title} completed!")
        except Exception as e:
            st.error(f"{title} failed: {e}")


with st.sidebar:
    html('<div class="logo"><i>🚦</i><div><b>AI Traffic</b><span>MONITORING</span></div></div>')
    st.write("")
    page = st.radio("Navigation", list(PAGES), format_func=lambda p: f"{PAGES[p]}   {p}", label_visibility="collapsed", key="nav")
    st.write("")
    html('<div class="chip"><span class="dot"></span>Dashboard Online</div>')
    st.caption("Engine · YOLO + ByteTrack · v1.0")

now = datetime.now()
html(f"""
<div class="topbar">
  <div><div class="time">{now:%I:%M}<small>{now:%p}</small></div><div class="date">{now:%A | %d %B %Y}</div></div>
  <div class="pill"><span class="dot"></span>Active module &nbsp;·&nbsp; <b>{page}</b></div>
  <div style="display:flex;gap:12px;align-items:center"><div class="chip"><span class="dot"></span>AI ENGINE READY</div><div class="avatar">AI</div></div>
</div>""")


if page == "Overview":

    chips = "".join(f"<span>{CLASS_ICONS[n]} {n}</span>" for n in CLASS_NAMES.values())
    html(f"""
    <div class="b-card hero">
      <div>
        <div class="tag"><span class="dot"></span>SMART CITY · AI MONITORING</div>
        <h1>See every vehicle.<br>Understand every road.</h1>
        <p>Detect, track and count cars, motorcycles, buses and trucks from traffic video,
        then turn the numbers into a congestion report.</p>
        <div class="types">{chips}</div>
      </div>
      <div>{hero_art()}</div>
    </div>""")

    modules = [
        ("Vehicle Detection", "gr", "◉", "Detection", "Find and label every vehicle in each frame of the video."),
        ("Vehicle Tracking", "v", "⌖", "Tracking", "Give each vehicle a unique ID and follow it across frames."),
        ("Traffic Analytics", "o", "▥", "Analytics", "Count vehicles crossing the line and rate the traffic density."),
        ("Reports", "cy", "▤", "Reports", "Review the latest results and download them as a CSV file."),
    ]
    for col, (target, color, icon, title, text) in zip(st.columns(4, gap="medium"), modules):
        with col, st.container(key=f"card_mod_{title.lower()}"):
            html(f'<div class="mod"><div class="ico {color}">{icon}</div><h4>{title}</h4><p>{text}</p></div>')
            st.button(f"Open {title}  →", key=f"open_{title}", on_click=go, args=(target,), width="stretch")

    st.write("")
    saved = st.session_state.traffic_report
    if saved is None:
        st.info("No report yet. Run Traffic Analytics to generate your first traffic report.")
    else:
        html('<div class="ctitle">Latest traffic report</div>')
        for col, (label, key) in zip(st.columns(3, gap="medium"),
                                     [("Total Vehicles Crossed", "total"), ("Currently Visible", "visible"),
                                      ("Traffic Density", "density")]):
            col.metric(label, saved[key])

elif page == "Vehicle Detection":
    live_page("detect", "Vehicle Detection", "Detect and classify cars, motorcycles, buses and trucks using YOLO.",
              detect_vehicles, "Vehicles Detected")

elif page == "Vehicle Tracking":
    live_page("tracking", "Vehicle Tracking", "Follow every vehicle with a unique tracking ID using ByteTrack.",
              track_vehicles, "Currently Tracked Vehicles")

elif page == "Traffic Analytics":
    head("Traffic Analytics", "Count vehicles crossing the line and estimate traffic density.")
    start = st.button("▶  Start Traffic Analysis", key="analytics")

    left, right = st.columns([3, 1.15], gap="medium")
    with left, st.container(key="card_analytics_feed"):
        html('<div class="ctitle">● Live Feed · Analytics</div>')
        video = st.empty()
        video.markdown(feed("Press Start to begin traffic analysis"), unsafe_allow_html=True)
    with right:
        slots = {k: st.empty() for k in ("total", "visible", "density")}
        labels = {"total": "Total Vehicles Crossed", "visible": "Currently Visible", "density": "Traffic Density"}
        for k, s in slots.items():
            s.metric(labels[k], "—" if k == "density" else 0)

    if start:
        stats = None
        try:
            with st.spinner("Analysing traffic video..."):
                for frame, stats in count_vehicles():
                    video.image(frame, channels="BGR", width="stretch")
                    for k, s in slots.items():
                        s.metric(labels[k], stats[k])
                    time.sleep(0.02)

            if stats is None:
                st.warning("No video frames were processed.")
            else:
                st.success("Traffic analysis completed successfully!")
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
                    col.metric(name, n)
                st.write("")
                result_cards(cats, report, "analytics")
        except Exception as e:
            st.error(f"Traffic analysis failed: {e}")

elif page == "Reports":
    head("Traffic Analysis Reports", "View and download your most recent traffic report.")
    saved = st.session_state.traffic_report

    if saved is None:
        st.info("No report available yet. Please run Traffic Analytics first.")
    else:
        st.success("Traffic report generated successfully!")
        for col, (label, key) in zip(st.columns(3, gap="medium"),
                                     [("Total Vehicles Crossed", "total"), ("Currently Visible", "visible"),
                                      ("Traffic Density", "density")]):
            col.metric(label, saved[key])
        st.write("")
        result_cards(saved["categories"], saved["report_data"], "report")

st.write("")
st.caption("AI Traffic Monitoring System | Python • YOLO • OpenCV • ByteTrack • Streamlit")