#!/usr/bin/env python3
"""web/pong.src.html + core.js + models/final_model.bin(base64) -> web/pong.html (단일 파일, 오프라인 실행)"""
import base64, pathlib
r = pathlib.Path(__file__).resolve().parent
core = (r / "core.js").read_text()
model = base64.b64encode((r.parent / "models" / "final_model.bin").read_bytes()).decode()
html = (r / "pong.src.html").read_text().replace("__CORE__", core).replace("__MODEL_B64__", model)
(r / "pong.html").write_text(html)
print("wrote", r / "pong.html", len(html), "bytes")
