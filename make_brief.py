#!/usr/bin/env python
"""Compact, self-contained LLM brief (Chinese) for MV shot planning with ComfyUI/Wan.
Usage: python make_brief.py <outdir> [--fps 16] [--lang zh]   -> <outdir>/agent_brief.md + agent_brief.json
Uses analysis.json and comfy/shots.json (run export_comfy.py first for matching fps; otherwise frames are computed here)."""
import argparse, json, os, math

ZH_MOOD = {"relaxed": "放松", "gentle": "温柔", "dreamy": "梦幻", "nostalgic": "怀旧", "happy": "快乐", "bright": "明亮", "euphoric": "欢欣",
           "intimate": "亲密", "romantic": "浪漫", "sad": "伤感", "dark": "阴暗", "tense": "紧张", "aggressive": "激烈", "epic": "宏大",
           "mysterious": "神秘", "triumphant": "昂扬", "hopeful": "充满希望", "calm": "平静", "playful": "俏皮", "energetic": "有活力",
           "melancholic": "忧郁", "hopeless": "绝望", "defeated": "失落", "gloomy": "阴郁", "grounded": "朴实", "cold": "冷淡", "serious": "严肃",
           "futuristic": "未来感", "straightforward": "直白", "intimate ": "亲密"}
STYLE_POLES = {"futuristic", "straightforward", "grounded", "serious", "cold"}   # weak 'other pole' of CLAP pairs, rarely useful for MV
STEM_ZH = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其他乐器"}
def stems_zh(a): return "+".join(STEM_ZH.get(x, x) for x in a.split("+")) if a else ""
def beat_desc(beats):
    if len(beats) < 3: return str(beats)
    import statistics; d = statistics.median([b - a for a, b in zip(beats, beats[1:])])
    return f"第{beats[0]}帧起约每{d:g}帧一拍（共{len(beats)}拍）"
NEG = {"sad", "dark", "tense", "aggressive", "hopeless", "defeated", "gloomy", "cold", "melancholic"}
SYN = [{"happy", "euphoric", "bright", "playful"}, {"relaxed", "calm", "gentle"}, {"dreamy", "mysterious"}, {"sad", "melancholic", "gloomy", "hopeless", "defeated"}]
TREND = {"rising": "渐强（上扬）", "falling": "渐弱（回落）", "peak": "高点（爆发）", "steady": "平稳"}
PACE = {"slow": "慢", "medium": "中", "fast": "快"}
SEC = {"intro": "前奏", "verse": "主歌", "chorus": "副歌", "bridge": "桥段", "climax/bridge": "高潮/桥段", "outro": "尾奏"}

def energy_word(e):
    return "很弱/安静" if e < 0.25 else "较弱" if e < 0.45 else "中等" if e < 0.62 else "较强" if e < 0.8 else "很强"

def sec_zh(l): return SEC.get(l.rstrip("?"), l.rstrip("?")) + ("（推测）" if l.endswith("?") else "")

def arr_zh(a):
    m = [("vocal-only", "仅人声/清唱"), ("full band with vocals", "全乐队+人声"), ("vocals over band without drums", "人声+伴奏（无鼓）"),
         ("sparse: vocals", "稀疏：人声+少量伴奏"), ("instrumental groove", "纯器乐律动（有鼓）"), ("near-silence", "近乎静音"), ("instrumental", "纯器乐")]
    return next((z for k, z in m if a.startswith(k)), a)

def pick_moods(seg, valence):
    tags = seg["mood"].get("clap", {}).get("mood", [])
    out, used = [], []
    for t, p in tags:
        if t in STYLE_POLES: continue
        npos = sum(1 for x, _ in tags[:5] if x not in NEG and x not in STYLE_POLES)
        if t in NEG and (valence is None or valence > -0.25) and npos >= 2: continue   # contradicts a mostly positive/calm segment
        grp = next((g for g in SYN if t in g), {t})
        if any(grp is u for u in used): continue
        used.append(grp); out.append((t, p))
        if len(out) == 3: break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("outdir"); ap.add_argument("--fps", type=int, default=16); ap.add_argument("--lang", default="zh"); ap.add_argument("--full", action="store_true", help="embed the full field glossary (for use without the skill)")
    a = ap.parse_args(); fps = a.fps
    A = json.load(open(os.path.join(a.outdir, "analysis.json"), encoding="utf-8")); g = A["global"]
    sp = os.path.join(a.outdir, "comfy", "shots.json"); SH = None
    if os.path.exists(sp):
        j = json.load(open(sp, encoding="utf-8")); SH = {s["id"]: s for s in j["shots"]} if j["fps"] == fps else None
    feel = g.get("feel", {}); key = g["key"]; gm = g.get("mood", {})
    L = []; P = L.append
    # ---- (1) instructions
    P("# 音乐MV分镜规划简报（供LLM代理使用）\n")
    P("## 1. 角色与任务说明\n")
    P("你是一名MV导演兼AI视频（ComfyUI + Wan）分镜规划师。下面是一首歌的自动化音乐分析结果。请**以每一句歌词（以及歌词之间的器乐段）为单位**，输出一份分镜表：每个片段对应一个或多个镜头，画面内容、镜头运动和剪辑节奏要与音乐的能量、情绪和节拍相匹配，同时服从第4节「我的创意」中用户给出的故事与风格。\n")
    P("**输出要求（每个片段一行/一张卡）**：片段id、时间、镜头类型（远/中/近/特写）、画面描述、镜头运动、是否对口型、剪辑点（帧号）、给Wan的英文提示词（可参考提供的 wan_prompt 改写）、运动强度/去噪建议。保持叙事连贯：同一段落内视觉风格统一，段落切换（section-transition）时可换场景或色调。\n")
    if a.full:
        P("**字段说明**：")
        P(f"- `time`：秒。`frames`：在 {fps} fps 下的全局帧范围；卡片里的 beat/downbeat/accent 帧号都是**相对该片段起点**（第0帧=片段开始）。")
        P("- `energy`：0–1，按本曲归一化（0=全曲最安静，1=最响）。文字等级：很弱<0.25≤较弱<0.45≤中等<0.62≤较强<0.8≤很强。")
        P("- `trend`：片段内部能量走向：渐强（适合推近、升降机、加速）、渐弱（拉远、减速、淡出）、高点（快切、手持、甩镜）、平稳。")
        P("- `pace`：镜头节奏（慢/中/快），由能量、音符密度、鼓的活跃度综合得出（pace_score 0–1），**不是**单纯看BPM；本曲为半拍感时以体感BPM为准。")
        P("- `arrangement`：编曲状态（仅人声/稀疏/全乐队/纯器乐），稀疏人声段适合特写对口型，纯器乐段适合空镜、B-roll、演奏镜头。")
        P("- 拍点以“起始帧+间隔”给出（完整帧号见 agent_brief.json）。`beat`=每拍，`downbeat`=每小节第一拍（首选剪辑点），`accent`=明显的重音/打击点（适合闪切、灯光闪烁、动作卡点）。")
        P(f"- **Wan 片段限制**：每个生成片段帧数须为 4n+1，且 ≤81 帧（{fps} fps 下约 {81/fps:.1f} 秒）。较长的片段已预先按小节切成多个 clip（见 `clips`），生成后按 trim 裁掉多余帧。")
        P("- `mood`：模型推测的情绪标签（最多3个，已去除低置信度的矛盾标签），仅供参考；歌词含义和用户创意优先。")
        P("- 标记“（推测）”的段落名（主歌/副歌等）是算法猜测，可能不准，请结合歌词重复情况自行判断。\n")
    else:
        P(f"字段含义与镜头翻译规则见 skill **mv-music-brief**（SKILL.md）。速查：energy 0–1（本曲归一化）；帧号 @ {fps}fps，卡片内帧号相对片段起点；Wan 每段 4n+1 且 ≤81 帧；“（推测）”=算法猜测；用户创意（第4节）优先级最高。\n")
    # ---- (2) overview
    P("## 2. 歌曲概览\n")
    dur = g["duration"]
    P(f"- 时长：{dur:.1f} 秒（{int(dur // 60)}:{int(dur % 60):02d}），共 {len(A['segments'])} 个片段（{sum(s['type']=='lyric' for s in A['segments'])} 句歌词）")
    if feel.get("half_time"): P(f"- 速度：记谱 {feel['notated_bpm']} BPM，**体感为半拍 {feel['felt_bpm']} BPM**（抒情、舒缓的律动），剪辑请按体感节奏")
    else: P(f"- 速度：{g['bpm']:.0f} BPM")
    mzh = {"G": "G", "minor": "小调", "major": "大调"}
    P(f"- 调性：{key['key']} {mzh.get(key['mode'], key['mode'])}；歌词语言：{g.get('lyrics_language')}；人声占比 {g['vocal_ratio']:.0%}")
    if gm: P(f"- 整体情绪：效价(valence) {gm.get('valence')}、唤醒度(arousal) {gm.get('arousal')}（-1~1，接近0为中性/偏平静）。注意：自动情绪模型对抒情怀旧类歌曲不太可靠，以歌词为准。")
    P("\n**情绪/能量走向（按段落）**：\n")
    for i, s in enumerate(A["sections"]):
        segs = [x for x in A["segments"] if x["section_index"] == i]
        lyr = next((x["text"] for x in segs if x["type"] == "lyric"), None)
        moods = []
        for x in segs:
            for t, _ in pick_moods(x, x["mood"].get("valence")):
                if t not in moods: moods.append(t)
        trend = {"rising": "逐渐升温", "falling": "逐渐回落", "steady": "保持"}.get(s["trend"], s["trend"])
        P(f"- {s['start']:.0f}–{s['end']:.0f}s {sec_zh(s['label'])}：能量{energy_word(s['energy_mean'])}（{s['energy_mean']}），{trend}；"
          f"{stems_zh(s.get('arrangement',''))}；情绪 {'/'.join(ZH_MOOD.get(m, m) for m in moods[:3])}" + (f"；首句「{lyr}」" if lyr else ("（纯器乐）" if "vocals" not in s.get("arrangement", "") else "（无新歌词起句）")))
    pk = sorted(A["segments"], key=lambda x: -x["energy"]["mean"])[:3]
    P("\n**能量高点**：" + "；".join(f"{x['start']:.0f}s（片段{A['segments'].index(x):03d}" + (f"「{x['text']}」" if x.get('text') else "，器乐") + f"，能量{x['energy']['mean']}）" for x in sorted(pk, key=lambda x: x['start'])))
    P("")
    # ---- (3) cards
    P("## 3. 分段卡片\n")
    cards = []
    for i, s in enumerate(A["segments"]):
        sid = f"{i:03d}"; sh = SH.get(sid) if SH else None
        f0, f1 = (sh["frame_start"], sh["frame_end"]) if sh else (round(s["start"] * fps), round(s["end"] * fps))
        rel = lambda ts: [round(t * fps) - f0 for t in ts]
        beats = sh["beat_frames"] if sh else rel([t for t in A["beats"] if s["start"] <= t < s["end"]])
        downs = sh["downbeat_frames"] if sh else rel([e["t"] for e in s["events"] if e["type"] == "downbeat"])
        accs = sh["accent_frames"] if sh else rel([e["t"] for e in s["events"] if e["type"] == "accent"])
        clips = [(c["frames"], c["wan_frames"]) for c in sh["clips"]] if sh else None
        moods = pick_moods(s, s["mood"].get("valence"))
        trans = [e for e in s["events"] if e["type"] == "section-transition"]
        cam = s["camera_motion_hint"]; e = s["energy"]
        card = dict(id=sid, type=s["type"], time=[s["start"], s["end"]], frames=[f0, f1], lyric=s.get("text"), section=s["section"],
                    arrangement=s["arrangement"], energy=e["mean"], energy_max=e["max"], trend=e["trend"], pace=cam["pace"], pace_score=cam.get("pace_score"),
                    mood=[t for t, _ in moods], beat_frames=beats, downbeat_frames=downs, accent_frames=accs, clips=clips,
                    camera=cam["suggestions"], wan_prompt=sh["wan_prompt"] if sh else None,
                    motion_strength=sh["motion_strength_range"] if sh else None, denoise=sh["denoise_range"] if sh else None,
                    section_change=bool(trans))
        cards.append(card)
        head = f"「{s['text']}」" if s.get("text") else "【器乐段】"
        P(f"### {sid}｜{s['start']:.2f}–{s['end']:.2f}s｜帧 {f0}–{f1}（{f1-f0}帧）｜{sec_zh(s['section'])}｜{head}")
        desc = f"{arr_zh(s['arrangement'])}；能量{energy_word(e['mean'])}（均值{e['mean']}，峰值{e['max']}），走向{TREND[e['trend']]}；节奏{PACE[cam['pace']]}（{cam.get('pace_score')}）"
        if moods: desc += "；情绪：" + "、".join(f"{ZH_MOOD.get(t, t)}" for t, _ in moods)
        if trans: desc += "；**此处进入新段落**"
        P("- " + desc)
        P(f"- 拍点：{beat_desc(beats)}；小节首拍帧：{downs}" + (f"；重音帧：{accs}" if accs else ""))
        if clips: P("- Wan分段：" + "，".join(f"{a}帧→生成{b}帧" for a, b in clips))
        if sh: P(f"- 镜头建议（Wan提示词参考）：*{sh['wan_prompt']}*｜运动强度{sh['motion_strength_range'][0]}–{sh['motion_strength_range'][1]}，去噪{sh['denoise_range'][0]}–{sh['denoise_range'][1]}")
        else: P(f"- 镜头建议：{'; '.join(cam['suggestions'])}")
        P("")
    # ---- (4) user
    P("## 4. 我的创意\n")
    P("> （请在此填写：故事梗概/人物设定、视觉风格与参考、场景与道具、色调、必须出现或避免的元素、是否需要对口型、画幅比例、角色一致性要求等。）\n")
    P("- 故事：\n- 人物：\n- 风格/色调：\n- 场景：\n- 其他要求：\n")
    md = "\n".join(L)
    open(os.path.join(a.outdir, "agent_brief.md"), "w", encoding="utf-8").write(md)
    json.dump(dict(fps=fps, overview=dict(duration=dur, notated_bpm=g["bpm"], feel=feel, key=key, language=g.get("lyrics_language"), mood=gm,
                   sections=[dict(start=s["start"], end=s["end"], label=s["label"], energy=s["energy_mean"], trend=s["trend"]) for s in A["sections"]]),
               segments=cards, user_creative={"story": "", "characters": "", "style": "", "scenes": "", "notes": ""}),
              open(os.path.join(a.outdir, "agent_brief.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {a.outdir}/agent_brief.md ({len(md)} chars)")

if __name__ == "__main__": main()
