from __future__ import annotations

import html
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from user_study.study_engine import ATTRIBUTE_NAMES, METHODS, QuestionView, StudyEngine


HOST = os.environ.get("USER_STUDY_HOST", "0.0.0.0")
PORT = int(os.environ.get("USER_STUDY_PORT", "8765"))
LOG_PATH = Path(__file__).resolve().parent / "server.log"


def write_log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def emit(message: str) -> None:
    write_log(message)
    try:
        print(message, flush=True)
    except Exception:
        pass


emit("正在加载二手车数据，请稍等...")
ENGINE = StudyEngine()
emit(f"数据加载完成。服务启动后可访问: http://{HOST}:{PORT}")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6978;
      --line: #d9e0e8;
      --accent: #2855ad;
      --accent-2: #0b7a75;
      --bg: #f6f8fb;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      font-size: 16px;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: stretch;
    }}
    .item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #fff;
    }}
    .item h2 {{ display: flex; justify-content: space-between; gap: 12px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    .bar {{
      height: 9px;
      background: #e8eef8;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    button, .button {{
      appearance: none;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 12px 18px;
      font-size: 16px;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
    }}
    button.secondary, .button.secondary {{ background: var(--accent-2); }}
    button.muted, .button.muted {{ background: #687386; }}
    input[type="text"] {{
      width: 100%;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 16px;
    }}
    label {{ display: block; margin: 10px 0; }}
    .checkline {{ display: flex; gap: 8px; align-items: center; }}
    .scale {{
      display: grid;
      grid-template-columns: 1fr repeat(7, auto);
      gap: 10px;
      align-items: center;
      margin: 16px 0;
    }}
    .notice {{
      background: #edf5ff;
      border: 1px solid #cfe3ff;
      border-radius: 8px;
      padding: 14px;
      margin: 16px 0;
      color: #234;
    }}
    @media (max-width: 780px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .scale {{ grid-template-columns: 1fr repeat(7, auto); overflow-x: auto; }}
    }}
  </style>
</head>
<body><main>{body}</main></body>
</html>""".encode("utf-8")


def item_card(label: str, item) -> str:
    rows = []
    for name, value in zip(ATTRIBUTE_NAMES, item.display_values):
        rows.append(
            f"<tr><th>{esc(name)}</th><td>{esc(value)}</td></tr>"
        )
    return f"""
<section class="item">
  <h2>{esc(label)} <span class="meta">#{item.point_index}</span></h2>
  <div class="meta">类别：{esc(item.dataset_name)}</div>
  <p><strong>{esc(item.title_zh)}</strong></p>
  <table>{''.join(rows)}</table>
</section>
"""


def render_index() -> bytes:
    methods = "".join(
        f"<label class='checkline'><input type='checkbox' name='method' value='{esc(method)}' checked> {esc(method)}</label>"
        for method in METHODS
    )
    body = f"""
<div class="panel">
  <h1>二手车偏好交互实验</h1>
  <p>
    你将每次比较两个二手车候选，并选择更偏好的一个。
    所有属性都已经归一化处理，数值越大代表越好。
    系统只记录你的选择、回答时间、问题数量和问卷评分。
  </p>
  <form method="post" action="/start">
    <label>用户编号
      <input type="text" name="participant_id" placeholder="例如：U001" required>
    </label>
    <h2>实验方法</h2>
    {methods}
    <div class="actions"><button type="submit">开始实验</button></div>
  </form>
</div>
"""
    return page("二手车偏好交互实验", body)


def render_question(session_id: str, question: QuestionView) -> bytes:
    progress = f"任务 {question.task_index}/{question.task_total}，第 {question.question_index} 个问题"
    session = ENGINE.get_session(session_id)
    undo_html = ""
    if ENGINE.can_undo(session):
        undo_html = f"""
  <form method="post" action="/undo" class="actions">
    <input type="hidden" name="sid" value="{esc(session_id)}">
    <button class="muted" type="submit">返回上一步</button>
  </form>
"""
    body = f"""
<div class="panel">
  <h1>{esc(question.method)}</h1>
  <p>{esc(progress)}。请选择你更喜欢的车辆。</p>
  <div class="notice">
    当前候选数量：{question.candidate_count}
  </div>
  <div class="grid">
    {item_card("车辆 A", question.item_a)}
    {item_card("车辆 B", question.item_b)}
  </div>
  <form method="post" action="/answer" class="actions">
    <input type="hidden" name="sid" value="{esc(session_id)}">
    <button type="submit" name="choice" value="A">我更喜欢 A</button>
    <button class="secondary" type="submit" name="choice" value="B">我更喜欢 B</button>
  </form>
  {undo_html}
</div>
"""
    return page("问题", body)


def render_survey(session_id: str) -> bytes:
    session = ENGINE.get_session(session_id)
    state = session.method_state if session else None
    if state is None:
        return render_finish(session_id)
    undo_html = ""
    if ENGINE.can_undo(session):
        undo_html = f"""
  <form method="post" action="/undo" class="actions">
    <input type="hidden" name="sid" value="{esc(session_id)}">
    <button class="muted" type="submit">返回上一步重新选择</button>
  </form>
"""
    recs = ENGINE.final_recommendations(state)
    rec_html = "".join(
        f"<li>{esc(item.dataset_name)} 车辆 #{item.point_index}：{esc(item.title_zh)}；"
        + ", ".join(f"{name}={value}" for name, value in zip(ATTRIBUTE_NAMES, item.display_values))
        + "</li>"
        for item in recs
    )
    scale_names = [
        ("satisfaction", "对推荐结果的满意度（越大越满意）。"),
        ("annoyance", "问题数量是否使你感到厌烦（越大越厌烦）。"),
    ]
    scales = []
    for key, text in scale_names:
        radios = "".join(
            f"<label><input type='radio' name='{key}' value='{i}' required> {i}</label>"
            for i in range(1, 8)
        )
        scales.append(f"<div class='scale'><div>{esc(text)}</div>{radios}</div>")
    body = f"""
<div class="panel">
  <h1>问卷：{esc(state.method)}</h1>
  <p>该方法共询问了 {state.questions} 个问题。请根据 1 到 7 分进行评价。</p>
  <div class="notice"><strong>所有类别的最终推荐：</strong><ol>{rec_html}</ol></div>
  <form method="post" action="/survey">
    <input type="hidden" name="sid" value="{esc(session_id)}">
    {''.join(scales)}
    <div class="actions"><button type="submit">提交并继续</button></div>
  </form>
  {undo_html}
</div>
"""
    return page("问卷", body)


def render_finish(session_id: str) -> bytes:
    session = ENGINE.get_session(session_id)
    participant = session.participant_id if session else ""
    body = f"""
<div class="panel">
  <h1>实验完成</h1>
  <p>感谢你，{esc(participant)}。你的回答已经保存。</p>
  <p>结果文件：</p>
  <ul>
    <li><code>code_1/user_study/results/interactions.csv</code></li>
    <li><code>code_1/user_study/results/surveys.csv</code></li>
  </ul>
  <a class="button" href="/">开始下一位用户</a>
</div>
"""
    return page("完成", body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self.send_html(render_index())
            return
        if parsed.path == "/question":
            sid = query.get("sid", [""])[0]
            session = ENGINE.get_session(sid)
            if session is None:
                self.redirect("/")
                return
            if ENGINE.is_complete(session):
                self.send_html(render_finish(sid))
                return
            question = ENGINE.current_question(session)
            if question is None:
                self.redirect("/survey?" + urlencode({"sid": sid}))
                return
            self.send_html(render_question(sid, question))
            return
        if parsed.path == "/survey":
            sid = query.get("sid", [""])[0]
            self.send_html(render_survey(sid))
            return
        if parsed.path == "/finish":
            sid = query.get("sid", [""])[0]
            self.send_html(render_finish(sid))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw)
        if self.path == "/start":
            participant_id = form.get("participant_id", ["anonymous"])[0]
            methods = form.get("method", [])
            session = ENGINE.create_session(participant_id, methods, True)
            self.redirect("/question?" + urlencode({"sid": session.session_id}))
            return
        if self.path == "/answer":
            sid = form.get("sid", [""])[0]
            choice = form.get("choice", ["A"])[0]
            session = ENGINE.get_session(sid)
            if session is not None:
                ENGINE.answer(session, choice)
            self.redirect("/question?" + urlencode({"sid": sid}))
            return
        if self.path == "/undo":
            sid = form.get("sid", [""])[0]
            session = ENGINE.get_session(sid)
            if session is not None:
                ENGINE.undo(session)
            self.redirect("/question?" + urlencode({"sid": sid}))
            return
        if self.path == "/survey":
            sid = form.get("sid", [""])[0]
            session = ENGINE.get_session(sid)
            survey = {}
            for key in ["satisfaction", "annoyance"]:
                try:
                    survey[key] = int(form.get(key, [""])[0])
                except ValueError:
                    survey[key] = ""
            if session is not None:
                ENGINE.finish_survey(session, survey)
            if session is not None and ENGINE.is_complete(session):
                self.redirect("/finish?" + urlencode({"sid": sid}))
            else:
                self.redirect("/question?" + urlencode({"sid": sid}))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_html(self, content: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        write_log("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    emit(f"User study 服务已启动，可以打开网页开始交互: http://{HOST}:{PORT}")
    emit(f"数据集目录: {ENGINE.data.dataset_dir}")
    emit(f"结果目录: {ENGINE.result_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
