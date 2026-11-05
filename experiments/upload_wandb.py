"""exp1 의 *_log.csv + results.txt 를 W&B 로 업로드 (run 1개 = 설정 1칸 x 시드 1개)"""
import csv, glob, re, os, wandb
PROJECT = os.environ.get("WB_PROJECT", "ds-pong-dqn")
final = {}
if os.path.exists("results.txt"):
    for l in open("results.txt"):
        m = re.search(r"tag=(\S+) gamma=([\d.]+) hit_r=([\d.]+) seed=(\d+)\s+final_win=([\d.]+) final_hits=([\d.]+)\s+best_eval=([\d.]+)", l)
        if m: final[m[1]] = dict(final_win=float(m[5]), final_hits=float(m[6]), best_eval=float(m[7]))
for f in sorted(glob.glob("g*_log.csv")):
    tag = f[:-8]
    m = re.match(r"g([\d.]+)_h([\d.]+)_s(\d+)", tag)
    g, h, s = float(m[1]), float(m[2]), int(m[3])
    run = wandb.init(project=PROJECT, name=tag, group=f"gamma{g}_hit{h}",
                     config=dict(gamma=g, hit_reward=h, seed=s, steps=300000, lr=3e-4, batch=32,
                                 target_sync=1000, replay=50000, eval_points=1000, lang="C", node="arm-cpu"),
                     tags=["exp1", "pong-dqn", "gamma-hit-grid"], reinit=True)
    for r in csv.DictReader(open(f)):
        wandb.log({"eval_win": float(r["eval_win"]), "eval_hits": float(r["eval_hits"]), "eps": float(r["eps"])}, step=int(r["step"]))
    if tag in final:
        for k, v in final[tag].items(): wandb.summary[k] = v
    print(tag, run.url)
    run.finish()
