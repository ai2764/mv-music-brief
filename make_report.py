#!/usr/bin/env python
"""Self-contained HTML report (Chinese UI): python make_report.py <outdir> [--audio <path-or-relative-url>] [--copy-audio]
Needs analysis.json; uses comfy/curves.csv and agent_brief.json when present."""
import argparse, csv, json, os, shutil
import make_brief as MB

def smooth(v, w=5):
    h = w // 2; return [round(sum(v[max(0, i - h):i + h + 1]) / len(v[max(0, i - h):i + h + 1]), 3) for i in range(len(v))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("outdir"); ap.add_argument("--audio"); ap.add_argument("--copy-audio", action="store_true")
    a = ap.parse_args(); od = a.outdir
    A = json.load(open(os.path.join(od, "analysis.json"), encoding="utf-8")); g = A["global"]
    B = json.load(open(os.path.join(od, "agent_brief.json"), encoding="utf-8")) if os.path.exists(os.path.join(od, "agent_brief.json")) else None
    # audio: relative path from report to audio file (optional copy next to report)
    src = a.audio or A["file"]; audio = None
    if a.copy_audio and os.path.exists(src):
        dst = os.path.join(od, "audio" + os.path.splitext(src)[1]); shutil.copy(src, dst); audio = os.path.basename(dst)
    elif a.audio and not os.path.exists(a.audio): audio = a.audio  # URL / relative given verbatim
    elif os.path.exists(src): audio = os.path.relpath(src, od)
    bands = {}
    cp = os.path.join(od, "comfy", "curves.csv")
    if os.path.exists(cp):
        rows = list(csv.DictReader(open(cp, encoding="utf-8", newline=""))); fps = 1 / (float(rows[1]["time"]) - float(rows[0]["time"])); step = max(1, round(fps / 8))
        for k in ["kick", "vocal", "bass_stem", "high", "sub_low", "rms"]:
            if k in rows[0]: bands[k] = [round(float(r[k]), 3) for r in rows[::step]]
        bands["_dt"] = step / fps
    cards = {c["id"]: c for c in B["segments"]} if B else {}
    segs = []
    for i, s in enumerate(A["segments"]):
        c = cards.get(f"{i:03d}", {})
        segs.append(dict(id=f"{i:03d}", t0=s["start"], t1=s["end"], type=s["type"], text=s.get("text"), sec=MB.sec_zh(s["section"]), seclab=s["section"].rstrip("?"),
                         e=s["energy"]["mean"], tr=s["energy"]["trend"], arr=MB.arr_zh(s["arrangement"]), pace=s["camera_motion_hint"]["pace"],
                         mood=[MB.ZH_MOOD.get(t, t) for t, _ in MB.pick_moods(s, s["mood"].get("valence"))],
                         cam="; ".join(s["camera_motion_hint"]["suggestions"]), wan=c.get("wan_prompt"),
                         clips=c.get("clips"), frames=c.get("frames")))
    feel = g.get("feel", {})
    arc = []
    for i, s in enumerate(A["sections"]):
        arc.append(dict(t0=s["start"], t1=s["end"], lab=s["label"].rstrip("?"), zh=MB.sec_zh(s["label"]), e=s["energy_mean"], tr=s["trend"]))
    D = dict(name=os.path.basename(A["file"]), dur=g["duration"], bpm=g["bpm"], feel=feel, key=f"{g['key']['key']} {'小调' if g['key']['mode']=='minor' else '大调'}",
             lang=g.get("lyrics_language"), vocal=g["vocal_ratio"], mood=g.get("mood", {}), sections=arc, segs=segs,
             energy=dict(t=A["curves_0.5s"]["t"], v=smooth(A["curves_0.5s"]["energy"])), downbeats=A["downbeats"], bands=bands, audio=audio,
             fps=B["fps"] if B else None)
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(D, ensure_ascii=False))
    p = os.path.join(od, "report.html"); open(p, "w", encoding="utf-8").write(html); print("wrote", p, f"({len(html)//1024} KB)")

TEMPLATE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MV 音乐分析报告</title><style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8e8ea;--mut:#9aa0aa;--acc:#ff6b6b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1300px;margin:auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}h2{font-size:16px;margin:18px 0 8px}
.card{background:var(--card);border-radius:10px;padding:14px;margin-bottom:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kv b{display:block;font-size:18px}.kv span{color:var(--mut);font-size:12px}.mut{color:var(--mut)}
canvas{width:100%;display:block;cursor:pointer}#tl{height:190px}#bands{height:240px}
.sticky{position:sticky;top:0;z-index:5;background:var(--bg);padding-top:6px}
#now{font-size:18px;min-height:28px;margin:6px 0;color:#ffd166}audio{width:100%}
table{width:100%;border-collapse:collapse}th,td{padding:6px 8px;border-bottom:1px solid #262a33;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:500;position:sticky;top:0;background:var(--card)}
tr.row{cursor:pointer}tr.row:hover{background:#20242d}tr.cur{background:#2c2433!important;outline:1px solid var(--acc)}
.bar{height:8px;background:#2a2f39;border-radius:4px;width:80px}.bar i{display:block;height:100%;border-radius:4px;background:linear-gradient(90deg,#4cc9f0,#f72585)}
.chip{display:inline-block;padding:0 6px;margin:1px;border-radius:9px;background:#2a3140;font-size:12px}.inst{color:var(--mut);font-style:italic}
.cam{font-size:12px;color:#c8ccd4;max-width:360px}.wan{font-size:11px;color:var(--mut)}.legend span{display:inline-block;margin-right:10px;font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
</style></head><body><div class="wrap">
<h1>🎬 MV 音乐分析报告 <span class="mut" id="fname"></span></h1>
<div class="card"><div class="grid" id="ov"></div><p id="arc" class="mut" style="margin:10px 0 0"></p></div>
<div class="sticky"><div class="card" style="margin-bottom:6px">
<audio id="au" controls preload="metadata"></audio><div id="noaudio" class="mut" style="display:none">（未找到音频文件：把歌曲放在报告旁边并用 --copy-audio 重新生成，或用 --audio 指定相对路径）</div>
<div id="now"></div><canvas id="tl"></canvas><div class="legend" id="leg"></div></div></div>
<div class="card"><h2 style="margin-top:0">频段 / 声部曲线</h2><canvas id="bands"></canvas></div>
<div class="card"><h2 style="margin-top:0">逐句分镜表 <span class="mut">（点击行跳转播放）</span></h2>
<table><thead><tr><th>#</th><th>时间</th><th>歌词</th><th>段落</th><th>能量</th><th>走向</th><th>编曲</th><th>情绪</th><th>镜头建议</th><th>Wan 帧</th></tr></thead><tbody id="tb"></tbody></table></div>
<p class="mut">由 mv-music-brief 生成 · 段落标签带“推测”的为算法猜测</p></div>
<script>
const D=/*DATA*/null;
const COL={intro:"#4361ee",verse:"#4cc9f0",chorus:"#f72585",bridge:"#ffd166","climax/bridge":"#ff6b35",outro:"#7209b7"};
const TR={rising:"↗ 渐强",falling:"↘ 渐弱",peak:"⏫ 高点",steady:"→ 平稳"};const PACE={slow:"慢",medium:"中",fast:"快"};
const $=id=>document.getElementById(id);const fmt=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,"0")}`;
$("fname").textContent=D.name;
const f=D.feel||{};const ov=[["时长",fmt(D.dur)],["速度",f.half_time?`体感 ${f.felt_bpm} BPM`:`${Math.round(D.bpm)} BPM`],["记谱BPM",D.bpm.toFixed(1)],["调性",D.key],["语言",D.lang||"-"],["人声占比",Math.round(D.vocal*100)+"%"],
 ["效价 valence",D.mood.valence??"-"],["唤醒 arousal",D.mood.arousal??"-"],["片段数",D.segs.length]];
$("ov").innerHTML=ov.map(([k,v])=>`<div class="kv"><b>${v}</b><span>${k}</span></div>`).join("");
const ew=e=>e<.25?"很安静":e<.45?"较弱":e<.62?"中等":e<.8?"较强":"很强";
const pk=[...D.segs].sort((a,b)=>b.e-a.e).slice(0,3).sort((a,b)=>a.t0-b.t0);
$("arc").innerHTML="<b>情绪/能量走向：</b>"+D.sections.map(s=>`${fmt(s.t0)} ${s.zh} ${ew(s.e)}${s.tr=="rising"?"↗":s.tr=="falling"?"↘":""}`).join(" → ")+
 "<br><b>能量高点：</b>"+pk.map(s=>`${fmt(s.t0)}${s.text?"「"+s.text+"」":"（器乐）"}`).join("，");
$("leg").innerHTML=Object.entries(COL).map(([k,c])=>`<span><i style="background:${c}"></i>${{intro:"前奏",verse:"主歌",chorus:"副歌",bridge:"桥段","climax/bridge":"高潮/桥段",outro:"尾奏"}[k]}</span>`).join("")+`<span><i style="background:#fff"></i>歌词</span><span class="mut">灰线=小节首拍</span>`;
const au=$("au");if(D.audio){au.src=D.audio;au.onerror=()=>{au.style.display="none";$("noaudio").style.display="block"}}else{au.style.display="none";$("noaudio").style.display="block"}
let T=0;const now=()=>D.audio&&!au.paused||au.currentTime?au.currentTime:T;
function setup(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const x=c.getContext("2d");x.setTransform(d,0,0,d,0,0);return [x,r.width,r.height]}
function drawTL(){const c=$("tl");const [x,W,H]=setup(c);const sx=t=>t/D.dur*W;x.clearRect(0,0,W,H);
 D.sections.forEach(s=>{x.fillStyle=(COL[s.lab]||"#888")+"33";x.fillRect(sx(s.t0),0,sx(s.t1)-sx(s.t0),H-22);x.fillStyle=COL[s.lab]||"#888";x.font="11px sans-serif";x.fillText(s.zh.replace("（推测）","?"),sx(s.t0)+3,12)});
 x.strokeStyle="#ffffff22";D.downbeats.forEach(t=>{x.beginPath();x.moveTo(sx(t),H-22);x.lineTo(sx(t),H-16);x.stroke()});
 x.beginPath();x.strokeStyle="#4cc9f0";x.lineWidth=1.5;D.energy.t.forEach((t,i)=>{const y=(H-26)-(D.energy.v[i])*(H-46)+4;i?x.lineTo(sx(t),y):x.moveTo(sx(t),y)});x.stroke();
 D.segs.forEach(s=>{if(s.type=="lyric"){x.fillStyle="#ffffffaa";x.fillRect(sx(s.t0),H-14,Math.max(1,sx(s.t1)-sx(s.t0)-1),8)}});
 const cur=cs();if(cur){x.fillStyle="#ff6b6b55";x.fillRect(sx(cur.t0),0,sx(cur.t1)-sx(cur.t0),H)}
 x.fillStyle="#9aa0aa";x.font="10px sans-serif";for(let t=0;t<D.dur;t+=30)x.fillText(fmt(t),sx(t)+2,H-1);
 x.strokeStyle="#ff6b6b";x.lineWidth=2;x.beginPath();x.moveTo(sx(now()),0);x.lineTo(sx(now()),H);x.stroke()}
const BL={kick:["底鼓 kick","#f72585"],vocal:["人声 vocal","#ffd166"],bass_stem:["贝斯 bass","#4361ee"],high:["高频 high","#4cc9f0"]};
function drawBands(){const c=$("bands");const [x,W,H]=setup(c);x.clearRect(0,0,W,H);const ks=Object.keys(BL).filter(k=>D.bands[k]);if(!ks.length){x.fillStyle="#888";x.fillText("（无 comfy/curves.csv，先运行 export_comfy.py）",10,20);return}
 const h=H/ks.length,dt=D.bands._dt,sx=t=>t/D.dur*W;ks.forEach((k,j)=>{const v=D.bands[k],y0=j*h;x.fillStyle=BL[k][1]+"55";x.strokeStyle=BL[k][1];x.beginPath();x.moveTo(0,y0+h);
  v.forEach((a,i)=>x.lineTo(sx(i*dt),y0+h-a*(h-4)));x.lineTo(W,y0+h);x.closePath();x.fill();x.fillStyle="#e8e8ea";x.font="11px sans-serif";x.fillText(BL[k][0],4,y0+12)});
 x.strokeStyle="#ff6b6b";x.lineWidth=2;x.beginPath();x.moveTo(sx(now()),0);x.lineTo(sx(now()),H);x.stroke()}
function cs(){const t=now();return D.segs.find(s=>t>=s.t0&&t<s.t1)}
$("tb").innerHTML=D.segs.map((s,i)=>`<tr class="row" id="r${i}" data-t="${s.t0}"><td>${s.id}</td><td>${fmt(s.t0)}–${fmt(s.t1)}<br><span class="mut">${(s.t1-s.t0).toFixed(1)}s</span></td>
<td>${s.text?s.text:'<span class="inst">【器乐】</span>'}</td><td>${s.sec}</td><td><div class="bar"><i style="width:${Math.round(s.e*100)}%"></i></div><span class="mut">${s.e.toFixed(2)} · 节奏${PACE[s.pace]||s.pace}</span></td>
<td>${TR[s.tr]||s.tr}</td><td>${s.arr}</td><td>${s.mood.map(m=>`<span class="chip">${m}</span>`).join("")}</td>
<td class="cam">${s.cam}${s.wan?`<div class="wan">Wan: ${s.wan}</div>`:""}</td><td>${s.frames?`${s.frames[1]-s.frames[0]}f`:""}${s.clips?"<br><span class=mut>"+s.clips.map(c=>c[1]).join("+")+"</span>":""}</td></tr>`).join("");
document.querySelectorAll("tr.row").forEach(r=>r.onclick=()=>seek(+r.dataset.t));
function seek(t){T=t;if(D.audio&&au.style.display!="none"){au.currentTime=t;au.play()}tick(true)}
[$("tl"),$("bands")].forEach(c=>c.onclick=e=>{const r=c.getBoundingClientRect();seek((e.clientX-r.left)/r.width*D.dur)});
let last=-1;function tick(scroll){const s=cs();const i=s?D.segs.indexOf(s):-1;$("now").textContent=s?(s.text?"♪ "+s.text:"【器乐】")+"  · "+s.sec:"";
 if(i!=last){document.querySelectorAll("tr.cur").forEach(r=>r.classList.remove("cur"));if(i>=0){const r=$("r"+i);r.classList.add("cur");if(scroll||!au.paused)r.scrollIntoView({block:"center",behavior:"smooth"})}last=i}
 drawTL();drawBands()}
au.ontimeupdate=()=>tick(false);window.onresize=()=>tick(false);(function loop(){if(!au.paused)tick(false);requestAnimationFrame(loop)})();tick(false);
</script></body></html>"""

if __name__ == "__main__": main()
