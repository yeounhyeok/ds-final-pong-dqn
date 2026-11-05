/*
 * Pong + DQN  (순수 C, 외부 라이브러리 없음)
 * 자료구조 및 실습 기말프로젝트 프로토타입
 *
 *   build : gcc -O2 -o pong_dqn pong_dqn.c -lm
 *   usage : ./pong_dqn gradcheck          역전파 수치 검증
 *           ./pong_dqn train [steps] [seed]   학습 -> best.bin, train_log.csv
 *           ./pong_dqn eval best.bin      학습된 정책 평가
 *
 * 구성: 1) 설정  2) 난수  3) 신경망(MLP)  4) Adam  5) Pong 환경
 *       6) 리플레이 버퍼(원형 큐)  7) 평가  8) gradcheck  9) DQN 학습  10) main
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef REAL
#define REAL float      /* gradcheck 는 -DREAL=double 로 빌드 (float+ReLU 꺾임 오탐 방지) */
#endif


/* ================= 1. 설정 ================= */

/* 신경망 크기: 입력 6 -> 은닉 64 -> 은닉 64 -> 출력 3 (위/정지/아래의 Q값) */
#define IN   6
#define H    64
#define OUT  3

/* 학습 하이퍼파라미터 */
#define BATCH          32        /* 한 번에 학습할 샘플 수 */
#define WARMUP         2000      /* 버퍼가 이만큼 찰 때까지 학습 안 함 */
#define REPLAY_CAP     50000     /* 리플레이 버퍼 크기 */
#define TARGET_SYNC    1000      /* 타깃망 복사 주기 (step) */
static float GAMMA = 0.95f;      /* 할인율 (env GAMMA) */
static float HIT_R = 0.3f;       /* 받아침 보상 (env HIT_R) */
static const char *TAG = "x";    /* 출력 파일 접두어 (env TAG) */
#define LR             3e-4f     /* Adam 학습률 */
#define EPS_MIN        0.05f     /* 탐험 확률 하한 */
#define EPS_DECAY      0.4f      /* 전체 step 의 40% 동안 1.0 -> EPS_MIN */
#define MAX_EP_STEPS   500       /* 에피소드 최대 길이 */
#define EVAL_EVERY     20000
#define EVAL_POINTS    1000

/* Pong 물리 (좌표는 0~1) */
#define PADDLE_HALF    0.10f     /* 패들 절반 높이 */
#define PADDLE_SPEED   0.03f     /* 에이전트 패들 속도 */
#define OPP_SPEED      0.018f    /* 상대 패들 속도 (에이전트보다 느림) */
#define BALL_VX0       0.020f
#define BALL_VX_MAX    0.050f


/* ================= 2. 난수 (xorshift) ================= */

static unsigned long long rs = 88172645463325252ULL;

static unsigned rnd(void)
{
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (unsigned)(rs >> 11);
}
static REAL frand(void)                { return (rnd() & 0xFFFFFF) / 16777216.0f; }  /* [0,1) */
static REAL frange(REAL a, REAL b)     { return a + (b - a) * frand(); }
static REAL clampf(REAL x, REAL a, REAL b) { return x < a ? a : (x > b ? b : x); }


/* ================= 3. 신경망 (MLP) ================= */

/* 모든 가중치를 구조체 하나에 담는다. 전부 REAL 이라서
 * (REAL*)로 보면 길이 NP 의 배열 -> 복사/저장/Adam 이 반복문 하나로 끝난다. */
typedef struct {
    REAL W1[H * IN],  b1[H];     /* 층1: IN  -> H   */
    REAL W2[H * H],   b2[H];     /* 층2: H   -> H   */
    REAL W3[OUT * H], b3[OUT];   /* 층3: H   -> OUT */
} Net;
#define NP ((int)(sizeof(Net) / sizeof(REAL)))      /* 파라미터 개수 = 4803 */

/* He 초기화: 균등분포 [-a, a], a = sqrt(3) * sqrt(2/입력수) */
static void init_layer(REAL *W, int n, REAL fan_in)
{
    REAL s = sqrtf(2.0f / fan_in);
    for (int i = 0; i < n; i++) W[i] = (frand() * 2 - 1) * 1.732f * s;
}
static void net_init(Net *n)
{
    memset(n, 0, sizeof *n);                         /* bias 는 0 으로 */
    init_layer(n->W1, H * IN, IN);
    init_layer(n->W2, H * H,  H);
    REAL s3 = sqrtf(1.0f / H);                       /* 출력층은 분산 1/H */
    for (int i = 0; i < OUT * H; i++) n->W3[i] = (frand() * 2 - 1) * 1.732f * s3;
}

/* 한 층:  y = W x + b      (W 는 nout x nin 행렬을 1차원 배열로 저장) */
static void dense(const REAL *W, const REAL *b, const REAL *x, int nin, REAL *y, int nout)
{
    for (int o = 0; o < nout; o++) {
        REAL s = b[o];
        for (int i = 0; i < nin; i++) s += W[o * nin + i] * x[i];
        y[o] = s;
    }
}
static void relu(REAL *v, int n) { for (int i = 0; i < n; i++) if (v[i] < 0) v[i] = 0; }

/* 순전파.  x[IN] -> q[OUT].  h1,h2 는 역전파에 쓰려고 호출자가 보관 */
static void forward(const Net *n, const REAL *x, REAL *h1, REAL *h2, REAL *q)
{
    dense(n->W1, n->b1, x,  IN, h1, H);   relu(h1, H);
    dense(n->W2, n->b2, h1, H,  h2, H);   relu(h2, H);
    dense(n->W3, n->b3, h2, H,  q,  OUT);                 /* 출력층은 ReLU 없음 */
}

/* 한 층의 역전파.  dy = dL/d(출력)
 *   gW += dy * x^T,  gb += dy,  dx = W^T dy   (dx 가 NULL 이면 입력쪽 기울기 생략) */
static void dense_back(const REAL *W, const REAL *x, const REAL *dy, int nin, int nout,
                       REAL *gW, REAL *gb, REAL *dx)
{
    for (int o = 0; o < nout; o++) {
        gb[o] += dy[o];
        for (int i = 0; i < nin; i++) gW[o * nin + i] += dy[o] * x[i];
    }
    if (!dx) return;
    for (int i = 0; i < nin; i++) {
        REAL s = 0;
        for (int o = 0; o < nout; o++) s += dy[o] * W[o * nin + i];
        dx[i] = s;
    }
}
/* ReLU 의 미분: 순전파에서 0 이었던(꺼진) 뉴런은 기울기도 0 */
static void relu_mask(REAL *d, const REAL *h, int n) { for (int i = 0; i < n; i++) if (h[i] <= 0) d[i] = 0; }

/* 역전파: dq = dL/dq 가 주어지면 모든 파라미터의 기울기를 g 에 누적 */
static void backward(const Net *n, const REAL *x, const REAL *h1, const REAL *h2,
                     const REAL *dq, Net *g)
{
    REAL dh2[H], dh1[H];
    dense_back(n->W3, h2, dq,  H,  OUT, g->W3, g->b3, dh2);   relu_mask(dh2, h2, H);
    dense_back(n->W2, h1, dh2, H,  H,   g->W2, g->b2, dh1);   relu_mask(dh1, h1, H);
    dense_back(n->W1, x,  dh1, IN, H,   g->W1, g->b1, NULL);
}


/* ================= 4. Adam ================= */

typedef struct { Net m, v; int t; } Adam;      /* m: 1차 모멘트, v: 2차 모멘트 */

static void adam_step(Net *n, const Net *g, Adam *a, REAL lr)
{
    const REAL b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    a->t++;
    REAL c1 = 1 - powf(b1, a->t), c2 = 1 - powf(b2, a->t);    /* 초기 편향 보정 */
    REAL *p = (REAL *)n, *m = (REAL *)&a->m, *v = (REAL *)&a->v;
    const REAL *gr = (const REAL *)g;
    for (int i = 0; i < NP; i++) {
        m[i] = b1 * m[i] + (1 - b1) * gr[i];
        v[i] = b2 * v[i] + (1 - b2) * gr[i] * gr[i];
        p[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
    }
}


/* ================= 5. Pong 환경 =================
 * 에이전트 = 왼쪽 패들(학습), 상대 = 오른쪽 패들(공 y 를 따라가는 고정 규칙) */

typedef struct { REAL bx, by, bvx, bvy, py, oy; } Env;   /* 공 위치/속도, 내 패들 y, 상대 패들 y */

static void env_reset(Env *e)
{
    e->bx = 0.5f;  e->by = frange(0.2f, 0.8f);
    e->py = 0.5f;  e->oy = 0.5f;
    e->bvx = (rnd() & 1) ? BALL_VX0 : -BALL_VX0;
    e->bvy = frange(-0.015f, 0.015f);
}

/* 관측 벡터 6개 (속도는 크기를 맞추려고 x20) */
static void env_obs(const Env *e, REAL *s)
{
    s[0] = e->bx;  s[1] = e->by;  s[2] = e->bvx * 20;  s[3] = e->bvy * 20;
    s[4] = e->py;  s[5] = e->oy;
}

/* 패들에 맞았을 때: 맞은 위치(중심에서 얼마나 먼가)에 따라 공의 y 속도를 바꾼다 */
static REAL spin(REAL bvy, REAL by, REAL paddle_y)
{
    return clampf(bvy + 0.02f * (by - paddle_y) / PADDLE_HALF, -0.04f, 0.04f);
}

/* 한 step 진행. act: 0=위 1=정지 2=아래.  반환=보상
 * 보상: 받아침 +0.3 / 상대가 놓침 +1 / 내가 놓침 -1 */
static REAL env_step(Env *e, int act, int *done, int *hit, int *won)
{
    REAL r = 0;  *done = 0;  *hit = 0;  *won = 0;

    e->py = clampf(e->py + (act - 1) * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF);

    /* 상대: 공이 자기 쪽으로 올 때만, 데드존(0.01) 밖이면 공 높이를 따라간다 */
    REAL d = e->by - e->oy;
    if (e->bvx > 0 || e->bx > 0.5f) {
        if (d > 0.01f)       e->oy += OPP_SPEED;
        else if (d < -0.01f) e->oy -= OPP_SPEED;
    }
    e->oy = clampf(e->oy, PADDLE_HALF, 1 - PADDLE_HALF);

    /* 공 이동 + 위/아래 벽 반사 */
    e->bx += e->bvx;  e->by += e->bvy;
    if (e->by < 0) { e->by = -e->by;    e->bvy = -e->bvy; }
    if (e->by > 1) { e->by = 2 - e->by; e->bvy = -e->bvy; }

    if (e->bx <= 0) {                                      /* 왼쪽 끝: 내 차례 */
        if (fabsf(e->by - e->py) <= PADDLE_HALF) {
            e->bx  = -e->bx;
            e->bvx = fminf(-e->bvx * 1.04f, BALL_VX_MAX);  /* 받을 때마다 4% 가속 */
            e->bvy = spin(e->bvy, e->by, e->py);
            r = HIT_R;  *hit = 1;
        } else { r = -1; *done = 1; }
    } else if (e->bx >= 1) {                               /* 오른쪽 끝: 상대 차례 */
        if (fabsf(e->by - e->oy) <= PADDLE_HALF) {
            e->bx  = 2 - e->bx;
            e->bvx = -e->bvx;
            e->bvy = spin(e->bvy, e->by, e->oy);
        } else { r = 1; *done = 1; *won = 1; }
    }
    return r;
}


/* ================= 6. 리플레이 버퍼 (원형 큐) ================= */

typedef struct { REAL s[IN], s2[IN], r; int a, d; } Tr;       /* (상태, 행동, 보상, 다음 상태, 종료) */
typedef struct { Tr *buf; int cap, head, size; } Replay;

static void rp_init(Replay *q, int cap)
{
    q->buf = malloc(sizeof(Tr) * cap);
    q->cap = cap;  q->head = 0;  q->size = 0;
}
/* 가득 차면 head 가 한 바퀴 돌아 가장 오래된 데이터를 덮어쓴다 */
static void rp_push(Replay *q, const Tr *t)
{
    q->buf[q->head] = *t;
    q->head = (q->head + 1) % q->cap;
    if (q->size < q->cap) q->size++;
}
static const Tr *rp_sample(const Replay *q) { return &q->buf[rnd() % q->size]; }


/* ================= 7. 평가 ================= */

static int argmax(const REAL *q)
{
    int b = 0;
    for (int i = 1; i < OUT; i++) if (q[i] > q[b]) b = i;
    return b;
}

typedef struct { REAL win; REAL hits; } Eval;      /* 득점 승률, 점수당 받아친 횟수 */

/* 탐험 없이(greedy) points 점 플레이. random_policy=1 이면 무작위 정책(기준선) */
static Eval evaluate(const Net *n, int points, int random_policy)
{
    Env e;  int wins = 0, hits = 0;
    REAL s[IN], h1[H], h2[H], q[OUT];
    for (int p = 0; p < points; p++) {
        env_reset(&e);
        int done = 0, hit, won = 0, steps = 0;
        while (!done && steps < 2000) {
            env_obs(&e, s);
            int a;
            if (random_policy) a = rnd() % 3;
            else { forward(n, s, h1, h2, q);  a = argmax(q); }
            env_step(&e, a, &done, &hit, &won);
            hits += hit;  steps++;
        }
        wins += won;
    }
    Eval r = { (REAL)wins / points, (REAL)hits / points };
    return r;
}


/* ================= 8. gradcheck (해석적 기울기 vs 수치 미분) ================= */

static int gradcheck(void)
{
    Net n;  net_init(&n);
    REAL x[IN], h1[H], h2[H], q[OUT];
    for (int i = 0; i < IN; i++) x[i] = frange(-1, 1);
    int act = 1;  REAL tgt = 0.7f;

    /* 손실 L = 0.5 (q[act] - tgt)^2  ->  dL/dq[act] = q[act] - tgt */
    forward(&n, x, h1, h2, q);
    Net g;  memset(&g, 0, sizeof g);
    REAL dq[OUT] = { 0, 0, 0 };
    dq[act] = q[act] - tgt;
    backward(&n, x, h1, h2, dq, &g);

    REAL *p = (REAL *)&n;  const REAL *ga = (const REAL *)&g;
    double maxrel = 0;  int checked = 0;
    for (int i = 0; i < NP; i += 7) {                       /* 7개마다 1개 샘플링 */
        REAL old = p[i], eps = (sizeof(REAL) == 8 ? 1e-5 : 1e-2), qa[OUT];
        p[i] = old + eps;  forward(&n, x, h1, h2, qa);  double lp = 0.5 * (qa[act] - tgt) * (qa[act] - tgt);
        p[i] = old - eps;  forward(&n, x, h1, h2, qa);  double lm = 0.5 * (qa[act] - tgt) * (qa[act] - tgt);
        p[i] = old;
        double num = (lp - lm) / (2 * eps), an = ga[i];     /* 중심차분 vs 역전파 */
        double den = fabs(num) + fabs(an);
        if (den < 1e-6) continue;
        double rel = fabs(num - an) / den;
        if (rel > maxrel) maxrel = rel;
        checked++;
    }
    printf("gradcheck: %d params checked of %d, max relative error = %.2e  -> %s\n",
           checked, NP, maxrel, maxrel < 1e-2 ? "PASS" : "FAIL");
    return maxrel < 1e-2 ? 0 : 1;
}


/* ================= 9. DQN 학습 ================= */

/* 리플레이 버퍼에서 BATCH 개를 뽑아 평균 기울기 g 를 만든다.
 *   target = r + GAMMA * max_a' Q_target(s', a')      (종료면 target = r)
 *   손실   = Huber(Q(s,a) - target)  -> 기울기를 [-1,1] 로 잘라 이상치에 둔감하게 */
static void compute_batch_gradient(const Net *on, const Net *tg, const Replay *rp, Net *g)
{
    REAL h1[H], h2[H], q[OUT], t1[H], t2[H], tq[OUT];
    memset(g, 0, sizeof *g);
    for (int b = 0; b < BATCH; b++) {
        const Tr *x = rp_sample(rp);

        forward(on, x->s, h1, h2, q);                         /* 현재 망의 예측 */
        REAL target = x->r;
        if (!x->d) {
            forward(tg, x->s2, t1, t2, tq);                   /* 타깃망으로 다음 상태 평가 */
            target += GAMMA * fmaxf(tq[0], fmaxf(tq[1], tq[2]));
        }
        REAL dq[OUT] = { 0, 0, 0 };
        dq[x->a] = clampf(q[x->a] - target, -1, 1) / BATCH;   /* 고른 행동의 출력만 기울기 */
        backward(on, x->s, h1, h2, dq, g);
    }
}

static int train(long steps, unsigned long long seed)
{
    rs ^= seed * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 20; i++) rnd();

    Net on, tg;  net_init(&on);  tg = on;                     /* on: 학습망, tg: 타깃망(복사본) */
    Adam *ad = calloc(1, sizeof(Adam));
    Replay rp;  rp_init(&rp, REPLAY_CAP);
    Env e;  env_reset(&e);

    Eval base = evaluate(&on, EVAL_POINTS, 1);
    printf("baseline random policy: win=%.3f hits/point=%.2f\n", base.win, base.hits);
    char logname[128], bestname[128], finalname[128];
    snprintf(logname,128,"%s_log.csv",TAG); snprintf(bestname,128,"%s_best.bin",TAG); snprintf(finalname,128,"%s_final.bin",TAG);
    FILE *lg = fopen(logname, "w");
    fprintf(lg, "step,eps,eval_win,eval_hits,sec\n");

    REAL s[IN], h1[H], h2[H], q[OUT];
    REAL best = -1;  clock_t t0 = clock();  int ep_steps = 0;

    for (long t = 1; t <= steps; t++) {
        /* (1) 행동 선택: ε-greedy. ε 는 1.0 에서 EPS_MIN 까지 선형 감소 */
        REAL eps = fmaxf(EPS_MIN, 1.0f - (REAL)t / (EPS_DECAY * steps));
        env_obs(&e, s);
        int a;
        if (frand() < eps) a = rnd() % 3;
        else { forward(&on, s, h1, h2, q);  a = argmax(q); }

        /* (2) 환경 진행 + 경험 저장 */
        int done, hit, won;
        REAL r = env_step(&e, a, &done, &hit, &won);
        ep_steps++;
        Tr tr;
        memcpy(tr.s, s, sizeof s);  env_obs(&e, tr.s2);
        tr.a = a;  tr.r = r;  tr.d = done;
        rp_push(&rp, &tr);
        if (done || ep_steps >= MAX_EP_STEPS) { env_reset(&e);  ep_steps = 0; }

        /* (3) 학습: 버퍼가 어느 정도 찬 뒤부터 매 step */
        if (rp.size >= WARMUP) {
            Net g;
            compute_batch_gradient(&on, &tg, &rp, &g);
            adam_step(&on, &g, ad, LR);
        }

        /* (4) 타깃망 동기화: 구조체 통째로 복사 */
        if (t % TARGET_SYNC == 0) tg = on;

        /* (5) 주기적 평가 + 최고 기록 저장 */
        if (t % EVAL_EVERY == 0) {
            Eval ev = evaluate(&on, EVAL_POINTS, 0);
            double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("step %7ld eps=%.2f  win=%.3f hits/point=%.2f  (%.0fs)\n", t, eps, ev.win, ev.hits, sec);
            fprintf(lg, "%ld,%.3f,%.3f,%.3f,%.1f\n", t, eps, ev.win, ev.hits, sec);
            fflush(lg);
            REAL score = ev.win + 0.001f * ev.hits;           /* 승률 우선, 동률이면 받아침 */
            if (score > best) {
                best = score;
                FILE *f = fopen(bestname, "wb");  fwrite(&on, sizeof on, 1, f);  fclose(f);
            }
        }
    }
    fclose(lg);
    { FILE *f = fopen(finalname, "wb"); fwrite(&on, sizeof on, 1, f); fclose(f); }
    { Eval fe = evaluate(&on, 5000, 0); printf("FINAL tag=%s gamma=%.2f hit_r=%.2f seed=%llu  final_win=%.4f final_hits=%.2f  best_eval=%.3f\n", TAG, GAMMA, HIT_R, seed, fe.win, fe.hits, best); }
    return 0;
}


/* ================= 10. main ================= */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s gradcheck | train [steps] [seed] | eval weights.bin\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "gradcheck")) { rs ^= 12345;  return gradcheck(); }

    if (getenv("GAMMA")) GAMMA = atof(getenv("GAMMA"));
    if (getenv("HIT_R")) HIT_R = atof(getenv("HIT_R"));
    if (getenv("TAG")) TAG = getenv("TAG");
    if (!strcmp(argv[1], "train"))
        return train(argc > 2 ? atol(argv[2]) : 300000,
                     argc > 3 ? strtoull(argv[3], 0, 10) : 1);

    if (!strcmp(argv[1], "eval") && argc > 2) {
        Net n;
        FILE *f = fopen(argv[2], "rb");
        if (!f || fread(&n, sizeof n, 1, f) != 1) { perror("weights"); return 1; }
        fclose(f);
        Eval ev = evaluate(&n, 1000, 0), rn = evaluate(&n, 1000, 1);
        printf("trained: win=%.3f hits/point=%.2f | random: win=%.3f hits/point=%.2f\n",
               ev.win, ev.hits, rn.win, rn.hits);
        return 0;
    }
    return 1;
}
