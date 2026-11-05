"""jobs2.txt 의 각 학습 CSV 를 실시간(10초 주기)으로 W&B 에 스트리밍. 끝나면 FINAL 값을 summary 에 기록."""
import csv, os, re, time, wandb
PROJECT = "ds-pong-dqn"
jobs = []
for l in open("jobs2.txt"):
    tag, binn, steps, seed, envs = (l.rstrip("\n").split("|") + [""])[:5]
    cfg = dict(binary=binn, steps=int(steps), seed=int(seed), node="arm-cpu", lang="C", input_dim=7 if "pred" in binn else 6,
               lr=3e-4, gamma=0.95, hit_reward=0.3, target_sync=1000, ddqn=0)
    for kv in envs.split():
        k, v = kv.split("="); cfg[{"LR": "lr", "SYNC": "target_sync", "DDQN": "ddqn"}[k]] = float(v) if k == "LR" else int(v)
    jobs.append((tag, cfg))
runs, sent, done = {}, {}, set()
def final_line(tag):
    p = f"{tag}.out"
    if os.path.exists(p):
        for l in open(p):
            m = re.search(r"final_win=([\d.]+) final_hits=([\d.]+)\s+best_eval=([\d.]+)", l)
            if m: return dict(final_win=float(m[1]), final_hits=float(m[2]), best_eval=float(m[3]))
while len(done) < len(jobs):
    for tag, cfg in jobs:
        if tag in done: continue
        f = f"{tag}_log.csv"
        if not os.path.exists(f): continue
        if tag not in runs:
            runs[tag] = wandb.init(project=PROJECT, name=tag, group=tag.rsplit("_s", 1)[0], config=cfg,
                                   tags=["exp2", "pong-dqn", "realtime"], reinit="create_new"); sent[tag] = 0
        rows = list(csv.DictReader(open(f)))
        for r in rows[sent[tag]:]:
            runs[tag].log({"eval_win": float(r["eval_win"]), "eval_hits": float(r["eval_hits"]), "eps": float(r["eps"])}, step=int(r["step"]))
        sent[tag] = len(rows)
        fin = final_line(tag)
        if fin:
            for k, v in fin.items(): runs[tag].summary[k] = v
            runs[tag].finish(); done.add(tag); print("done", tag, fin, flush=True)
    time.sleep(10)
print("ALL STREAMED", flush=True)
