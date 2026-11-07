/*
 * pong_core.h — 게임 로직 (화면·입력과 분리). UI 는 터미널판 pong_game.c
 *
 *   - v2 모델(models/final_model.bin)을 불러와 왼쪽 패들(AI)을 조종
 *   - 난이도 1~3 = 같은 모델의 '착지 예측' 입력에 가우시안 오차 σ 를 섞음 (ranking 의 diff 와 같은 번호)
 *   - 사람 = 오른쪽 패들. 받아친 횟수를 세서 ranking.c 의 점수 계산에 넘긴다
 *
 * pong_dqn_v2.c 를 include 해서 신경망(forward)과 환경(env_reset/env_obs/spin)을 그대로 재사용한다.
 * 반드시 -DPRED 로 빌드 (v2 모델은 입력이 7개).
 */
#ifndef PONG_CORE_H
#define PONG_CORE_H

#ifndef PRED
#error "pong_core.h 는 -DPRED 로 빌드해야 합니다 (v2 모델 입력 7개)"
#endif

#define main pong_dqn_main          /* pong_dqn_v2.c 의 main 은 이름만 바꿔서 같이 컴파일 */
#include "pong_dqn_v2.c"
#undef main
#include "ranking.h"

#define MODEL_PATH_DEFAULT "models/final_model.bin"


/* ================= 난이도 / 설정 ================= */

typedef struct { const char *name, *desc; REAL sigma; int tracker_win; } Difficulty;

/* DIFFS[diff - 1].  tracker_win = 규칙 추적 상대 대비 AI 승률(%) — README / play selftest 측정값 */
static const Difficulty DIFFS[3] = {
    { "쉬움",   "AI 의 공 착지 예측에 큰 오차 (σ=0.30)",  0.30f, 29 },
    { "보통",   "AI 의 공 착지 예측에 작은 오차 (σ=0.15)", 0.15f, 69 },
    { "어려움", "오차 없음 - 학습된 그대로 (σ=0)",         0.00f, 99 },
};
#define N_DIFFS 3

static const int WIN_OPTS[] = { 3, 5, 7, 11 };                 /* 목표 점수 */
#define N_WIN_OPTS 4
static const struct { const char *name; int fps; } SPEED_OPTS[] = {   /* 1 프레임 = 1 step */
    { "느림", 20 }, { "보통", 30 }, { "빠름", 40 }, { "매우 빠름", 50 },
};
#define N_SPEED_OPTS 4

typedef struct {
    int diff;       /* 1~3 */
    int win_i;      /* WIN_OPTS 인덱스 */
    int speed_i;    /* SPEED_OPTS 인덱스 */
    int show_q;     /* AI 판단(Q값) 표시 */
} Settings;

static inline Settings settings_default(void)
{
    Settings s = { 2, 2, 1, 1 };    /* 보통, 7점, 보통 속도, Q값 표시 */
    return s;
}
static inline int clampi(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
static inline void settings_fix(Settings *s)
{
    s->diff    = clampi(s->diff, 1, N_DIFFS);
    s->win_i   = clampi(s->win_i, 0, N_WIN_OPTS - 1);
    s->speed_i = clampi(s->speed_i, 0, N_SPEED_OPTS - 1);
    s->show_q  = s->show_q ? 1 : 0;
}


/* ================= 모델 ================= */

static Net ai_net;

/* 성공 0. 실패하면 err 에 이유를 적고 1 */
static inline int load_model(const char *path, char *err, size_t errn)
{
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "모델 파일 '%s' 을 열 수 없습니다.", path);  return 1; }
    int ok = fread(&ai_net, sizeof ai_net, 1, f) == 1 && fgetc(f) == EOF;
    fclose(f);
    if (!ok) {
        snprintf(err, errn, "'%s' 의 크기가 맞지 않습니다 (v2 MLP %d-%d-%d-%d, %zu 바이트 기대).",
                 path, IN, H, H, OUT, sizeof ai_net);
        return 1;
    }
    return 0;
}

static inline void core_seed(void)          /* 실행마다 다른 서브 */
{
    rs ^= (unsigned long long)time(NULL) * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 20; i++) rnd();
}

static inline REAL gauss(void)               /* 표준정규분포 (Box-Muller) */
{
    REAL u = frand() + 1e-6f, v = frand();
    return sqrtf(-2 * logf(u)) * cosf(6.2831853f * v);
}


/* ================= 한 경기 ================= */

typedef struct {
    Env  e;                 /* 왼쪽 py = AI, 오른쪽 oy = 사람 */
    int  diff, win_score;
    int  score_ai, score_me;
    int  hits_me;           /* 사람이 받아친 횟수 (랭킹 점수용) */
    REAL q[OUT];            /* AI 가 본 Q값 (위/정지/아래) */
    int  ai_act;
    int  wait;              /* 서브 전 대기 프레임 */
    int  over;
    char msg[64];           /* 경기장 가운데 문구 (UTF-8, 따옴표 없음) */
} Game;

enum { EV_NONE, EV_AI_POINT, EV_MY_POINT, EV_GAME_OVER };

/* AI 행동: 관측 -> (난이도만큼 착지 예측에 오차) -> 순전파 -> Q 최대 행동 */
static inline int ai_decide(const Env *e, REAL sigma, REAL *q)
{
    REAL s[IN], h1[H], h2[H];
    env_obs(e, s);
    if (sigma > 0) s[6] = clampf(s[6] + sigma * gauss(), 0, 1);
    forward(&ai_net, s, h1, h2, q);
    return argmax(q);
}

/* 사람 vs AI 1 step. 물리는 env_step 과 같고 오른쪽 패들만 규칙 대신 사람.
 * act: AI (0 위 1 정지 2 아래), dir: 사람 (-1 위 0 정지 +1 아래)
 * 반환 0 진행 / 1 AI 득점 / 2 사람 득점.  *my_hit = 사람이 이번 step 에 받아쳤으면 1 */
static inline int human_step(Env *e, int act, int dir, int *my_hit)
{
    *my_hit = 0;
    e->py = clampf(e->py + (act - 1) * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF);
    e->oy = clampf(e->oy + dir * PADDLE_SPEED,       PADDLE_HALF, 1 - PADDLE_HALF);

    e->bx += e->bvx;  e->by += e->bvy;
    if (e->by < 0) { e->by = -e->by;    e->bvy = -e->bvy; }
    if (e->by > 1) { e->by = 2 - e->by; e->bvy = -e->bvy; }

    if (e->bx <= 0) {                                      /* 왼쪽 끝: AI 차례 */
        if (fabsf(e->by - e->py) > PADDLE_HALF) return 2;
        e->bx  = -e->bx;
        e->bvx = fminf(-e->bvx * 1.04f, BALL_VX_MAX);      /* 학습 때처럼 AI 가 받을 때만 가속 */
        e->bvy = spin(e->bvy, e->by, e->py);
    } else if (e->bx >= 1) {                               /* 오른쪽 끝: 사람 차례 */
        if (fabsf(e->by - e->oy) > PADDLE_HALF) return 1;
        e->bx  = 2 - e->bx;
        e->bvx = -e->bvx;
        e->bvy = spin(e->bvy, e->by, e->oy);
        *my_hit = 1;
    }
    return 0;
}

static inline void game_start(Game *g, int diff, int win_score, int fps)
{
    memset(g, 0, sizeof *g);
    g->diff = clampi(diff, 1, N_DIFFS);
    g->win_score = win_score;
    g->ai_act = 1;
    env_reset(&g->e);
    g->wait = fps;  strcpy(g->msg, "준비");
}

/* 한 프레임. dir = 사람 패들 방향. 반환 EV_* */
static inline int game_frame(Game *g, int dir, int fps)
{
    if (g->over) return EV_NONE;
    if (g->wait > 0) {
        if (--g->wait == 0) g->msg[0] = 0;
        return EV_NONE;
    }
    g->ai_act = ai_decide(&g->e, DIFFS[g->diff - 1].sigma, g->q);

    int hit, r = human_step(&g->e, g->ai_act, dir, &hit);
    g->hits_me += hit;
    if (!r) return EV_NONE;

    if (r == 1) g->score_ai++;
    else        g->score_me++;
    env_reset(&g->e);

    if (g->score_ai >= g->win_score || g->score_me >= g->win_score) {
        g->over = 1;
        snprintf(g->msg, sizeof g->msg, "%s  %d : %d", g->score_me > g->score_ai ? "당신의 승리!" : "AI 승리",
                 g->score_ai, g->score_me);
        return EV_GAME_OVER;
    }
    strcpy(g->msg, r == 1 ? "AI 득점" : "득점!");
    g->wait = fps * 4 / 5;
    return r == 1 ? EV_AI_POINT : EV_MY_POINT;
}

/* 이 경기의 랭킹 점수 (ranking.c 의 공식) */
static inline int game_rank_score(const Game *g)
{
    return rank_calc_score(g->score_me, g->hits_me, g->score_me > g->score_ai, g->diff);
}

/* 화면 폭: 한글 등 동아시아 전각 글자 2칸, 나머지(ASCII·박스 문자·블록 등) 1칸 */
static inline int text_width(const char *s)
{
    int w = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        unsigned cp;  int n;
        if (*p < 0x80)                { cp = *p;          n = 1; }
        else if ((*p & 0xE0) == 0xC0) { cp = *p & 0x1F;   n = 2; }
        else if ((*p & 0xF0) == 0xE0) { cp = *p & 0x0F;   n = 3; }
        else                          { cp = *p & 0x07;   n = 4; }
        for (int i = 1; i < n && p[i]; i++) cp = (cp << 6) | (p[i] & 0x3F);
        for (int i = 0; i < n && *p; i++) p++;
        if (cp >= 0x80 && cp < 0xA0) continue;
        w += (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0xA4CF) || (cp >= 0xAC00 && cp <= 0xD7A3)
          || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF00 && cp <= 0xFF60)
          || (cp >= 0xFFE0 && cp <= 0xFFE6) || cp >= 0x1F300 ? 2 : 1;
    }
    return w;
}

#endif
