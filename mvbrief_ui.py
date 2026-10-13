#!/usr/bin/env python
"""mvbrief-ui — Gradio drag-and-drop UI: audio (+ optional lyrics) -> zipped outputs + brief preview."""
import os, shutil, tempfile, time
import mvbrief_cli

def process(audio, lyrics_file, lyrics_text, fps, whisper_model):
    if not audio: raise ValueError("please upload an audio file")
    work = tempfile.mkdtemp(prefix="mvbrief_"); name = os.path.splitext(os.path.basename(audio))[0]
    out = os.path.join(work, name); lyr = None
    if lyrics_text and lyrics_text.strip():
        lyr = os.path.join(work, "lyrics.txt"); open(lyr, "w", encoding="utf-8").write(lyrics_text)
    elif lyrics_file: lyr = lyrics_file
    logs = []
    mvbrief_cli.run_pipeline(audio, out, lyr, int(fps), None, whisper_model or None, log=logs.append, copy_audio=True)
    shutil.rmtree(os.path.join(out, "stems"), ignore_errors=True) if os.environ.get("MVBRIEF_UI_NO_STEMS") else None
    z = shutil.make_archive(os.path.join(work, f"{name}_mvbrief"), "zip", out)
    brief = open(os.path.join(out, "agent_brief.md"), encoding="utf-8").read()
    return z, brief, os.path.join(out, "analysis.png"), "\n".join(logs)

def main():
    import gradio as gr
    with gr.Blocks(title="mvbrief") as ui:
        gr.Markdown("# mvbrief — 音乐 → MV 分镜简报\n拖入音频（可选歌词），生成 analysis / ComfyUI 曲线 / agent_brief，并打包下载。")
        with gr.Row():
            audio = gr.Audio(type="filepath", label="音频 Audio")
            with gr.Column():
                lf = gr.File(label="歌词 txt（可选）", type="filepath", file_types=[".txt"])
                lt = gr.Textbox(label="或直接粘贴歌词（每行一句）", lines=6)
                fps = gr.Dropdown([16, 24, 30], value=16, label="fps")
                wm = gr.Dropdown(["small", "medium", "large-v3"], value="small", label="Whisper model")
                btn = gr.Button("Analyze", variant="primary")
        z = gr.File(label="下载 zip"); img = gr.Image(label="energy / sections")
        brief = gr.Markdown(); log = gr.Textbox(label="log", lines=4)
        btn.click(process, [audio, lf, lt, fps, wm], [z, brief, img, log])
    ui.queue().launch(server_name=os.environ.get("MVBRIEF_HOST", "127.0.0.1"), server_port=int(os.environ.get("MVBRIEF_PORT", 7860)))

if __name__ == "__main__": main()
