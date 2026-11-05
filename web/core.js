/* core.js — pong_dqn_v2.c 의 환경·관측·신경망 추론을 JS 로 옮긴 것 (학습 코드 없음, 추론 전용).
 * 가중치 배열 순서: W1[64*7] b1[64] W2[64*64] b2[64] W3[3*64] b3[3] (float32, little-endian) */
const PADDLE_HALF = 0.10, PADDLE_SPEED = 0.03, OPP_SPEED = 0.018, BALL_VX0 = 0.020, BALL_VX_MAX = 0.050;
const IN = 7, H = 64, OUT = 3;
const clampf = (x, a, b) => x < a ? a : (x > b ? b : x);

function parseNet(arrayBuffer) {
  const f = new Float32Array(arrayBuffer);
  if (f.length !== H * IN + H + H * H + H + OUT * H + OUT) throw new Error('모델 크기 불일치: ' + f.length);
  let o = 0; const take = n => { const s = f.subarray(o, o + n); o += n; return s; };
  return { W1: take(H * IN), b1: take(H), W2: take(H * H), b2: take(H), W3: take(OUT * H), b3: take(OUT) };
}
function dense(W, b, x, nin, nout) {
  const y = new Float64Array(nout);
  for (let o = 0; o < nout; o++) { let s = b[o]; for (let i = 0; i < nin; i++) s += W[o * nin + i] * x[i]; y[o] = s; }
  return y;
}
function forward(n, x) {
  const h1 = dense(n.W1, n.b1, x, IN, H); for (let i = 0; i < H; i++) if (h1[i] < 0) h1[i] = 0;
  const h2 = dense(n.W2, n.b2, h1, H, H); for (let i = 0; i < H; i++) if (h2[i] < 0) h2[i] = 0;
  return dense(n.W3, n.b3, h2, H, OUT);
}
function envReset(rand) {
  return { bx: 0.5, by: 0.2 + 0.6 * rand(), py: 0.5, oy: 0.5,
           bvx: rand() < 0.5 ? BALL_VX0 : -BALL_VX0, bvy: -0.015 + 0.03 * rand() };
}
function landingY(e) {                       // 벽 반사를 접어서 내 쪽 도착 y (공이 멀어지면 0.5)
  if (!(e.bvx < 0)) return 0.5;
  const t = e.bx / (-e.bvx); let y = (e.by + e.bvy * t) % 2.0; if (y < 0) y += 2.0;
  return y > 1 ? 2.0 - y : y;
}
function envObs(e) { return [e.bx, e.by, e.bvx * 20, e.bvy * 20, e.py, e.oy, landingY(e)]; }
const spin = (bvy, by, py) => clampf(bvy + 0.02 * (by - py) / PADDLE_HALF, -0.04, 0.04);
function gauss(rand) { const u = rand() + 1e-6, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v); }

/* AI 행동. sigma>0 이면 착지 예측 입력에 가우시안 오차 (난이도). 0 위 / 1 정지 / 2 아래 */
function aiAct(net, e, sigma, rand) {
  const s = envObs(e);
  if (sigma > 0) s[6] = clampf(s[6] + sigma * gauss(rand), 0, 1);
  const q = forward(net, s); let a = 0; for (let i = 1; i < OUT; i++) if (q[i] > q[a]) a = i;
  return a;
}
/* 한 step. 반환 0=계속, 1=AI(왼쪽)가 놓침, 2=오른쪽이 놓침. ev.aiHit / ev.rightHit 에 받아침 표시 */
function step(e, aact, oact, ospeed, ev) {
  ev.aiHit = false; ev.rightHit = false;
  e.py = clampf(e.py + (aact - 1) * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF);
  e.oy = clampf(e.oy + (oact - 1) * ospeed, PADDLE_HALF, 1 - PADDLE_HALF);
  e.bx += e.bvx; e.by += e.bvy;
  if (e.by < 0) { e.by = -e.by; e.bvy = -e.bvy; }
  if (e.by > 1) { e.by = 2 - e.by; e.bvy = -e.bvy; }
  if (e.bx <= 0) {
    if (Math.abs(e.by - e.py) <= PADDLE_HALF) { e.bx = -e.bx; e.bvx = Math.min(-e.bvx * 1.04, BALL_VX_MAX); e.bvy = spin(e.bvy, e.by, e.py); ev.aiHit = true; }
    else return 1;
  } else if (e.bx >= 1) {
    if (Math.abs(e.by - e.oy) <= PADDLE_HALF) { e.bx = 2 - e.bx; e.bvx = -e.bvx; e.bvy = spin(e.bvy, e.by, e.oy); ev.rightHit = true; }
    else return 2;
  }
  return 0;
}
function trackerAct(e) {                      // 학습 환경의 규칙 추적 상대 (검증용)
  const d = e.by - e.oy;
  if (e.bvx > 0 || e.bx > 0.5) { if (d > 0.01) return 2; if (d < -0.01) return 0; }
  return 1;
}
const SIGMA = { 1: 0.30, 2: 0.15, 3: 0.0 };
if (typeof module !== 'undefined') module.exports = { parseNet, forward, envReset, envObs, aiAct, step, trackerAct, SIGMA, OPP_SPEED, PADDLE_SPEED, PADDLE_HALF };
