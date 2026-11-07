/* play.c — 터미널에서 DQN 과 Pong 대전.  빌드: make play
 *   ./play [난이도 1|2|3] [모델.bin]     사람=오른쪽 패들, DQN=왼쪽 패들
 *   ./play selftest [모델.bin]            (비대화형) 규칙 추적 상대와 붙여 승률 검증
 * 조작: w/↑ 위로, s/↓ 아래로 (누르는 동안 이동, 떼면 멈춤), 스페이스 즉시 정지, q 종료
 * 터미널은 키를 뗀 순간을 알 수 없으므로 '키 반복 입력이 끊기면 멈춤'으로 구현한다.
 *   HOLD0 = 첫 입력 후 유지 틱(키 반복 시작 전 지연을 메움), HOLD = 반복 입력마다 유지 틱 (환경변수로 조정)
 * 난이도는 학습된 같은 모델의 착지 예측 입력에 가우시안 오차 σ 를 섞어 만든다 (1=0.30, 2=0.15, 3=0). */
#define main orig_main
#include "pong_dqn_v2.c"
#undef main
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include "ranking.h"

#define COLS 61
#define ROWS 21
#define WIN_SCORE 7
#define TICK_US 30000

static struct termios old_tio;
static int raw_on = 0;
static void tty_restore(void)
{
    if (raw_on) { tcsetattr(0, TCSANOW, &old_tio); raw_on = 0; }
    printf("\x1b[?25h\x1b[0m\n");  fflush(stdout);
}
static void on_sig(int s) { (void)s; tty_restore(); _exit(0); }
static void tty_raw(void)
{
    tcgetattr(0, &old_tio);
    struct termios t = old_tio;
    t.c_lflag &= ~(ICANON | ECHO);  t.c_cc[VMIN] = 0;  t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &t);
    fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
    raw_on = 1;
    atexit(tty_restore);  signal(SIGINT, on_sig);  signal(SIGTERM, on_sig);
    printf("\x1b[2J\x1b[?25l");
}

static REAL gauss(void)
{
    REAL u = frand() + 1e-6f, v = frand();
    return sqrtf(-2 * logf(u)) * cosf(6.2831853f * v);
}

/* 한 step. aact=왼쪽(AI) 행동, oact=오른쪽 행동(0 위/1 정지/2 아래), ospeed=오른쪽 패들 속도.
 * 반환 0=계속, 1=AI 가 놓침(오른쪽 득점), 2=오른쪽이 놓침(AI 득점). 물리는 env_step 과 동일. */
static int step2(Env *e, int aact, int oact, REAL ospeed, int *ai_hit)
{
    *ai_hit = 0;
    e->py = clampf(e->py + (aact - 1) * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF);
    e->oy = clampf(e->oy + (oact - 1) * ospeed,        PADDLE_HALF, 1 - PADDLE_HALF);
    e->bx += e->bvx;  e->by += e->bvy;
    if (e->by < 0) { e->by = -e->by;    e->bvy = -e->bvy; }
    if (e->by > 1) { e->by = 2 - e->by; e->bvy = -e->bvy; }
    if (e->bx <= 0) {
        if (fabsf(e->by - e->py) <= PADDLE_HALF) {
            e->bx = -e->bx;  e->bvx = fminf(-e->bvx * 1.04f, BALL_VX_MAX);
            e->bvy = spin(e->bvy, e->by, e->py);  *ai_hit = 1;
        } else return 1;
    } else if (e->bx >= 1) {
        if (fabsf(e->by - e->oy) <= PADDLE_HALF) {
            e->bx = 2 - e->bx;  e->bvx = -e->bvx;  e->bvy = spin(e->bvy, e->by, e->oy);
        } else return 2;
    }
    return 0;
}

static int ai_act(const Net *n, const Env *e, REAL sigma)
{
    REAL s[IN], h1[H], h2[H], q[OUT];
    env_obs(e, s);
#ifdef PRED
    if (sigma > 0) s[6] = clampf(s[6] + sigma * gauss(), 0, 1);
#endif
    forward(n, s, h1, h2, q);
    return argmax(q);
}

static int tracker_act(const Env *e)           /* 학습 환경의 상대 규칙 그대로 */
{
    REAL d = e->by - e->oy;
    if (e->bvx > 0 || e->bx > 0.5f) { if (d > 0.01f) return 2; if (d < -0.01f) return 0; }
    return 1;
}

static void draw(const Env *e, int sa, int sh, const char *msg, int diff)
{
    static char g[ROWS][COLS + 1];
    for (int r = 0; r < ROWS; r++) { memset(g[r], ' ', COLS); g[r][COLS] = 0; g[r][COLS / 2] = ':'; }
    for (int r = 0; r < ROWS; r++) {
        REAL y = (REAL)r / (ROWS - 1);
        if (fabsf(y - e->py) <= PADDLE_HALF) g[r][0] = '#';
        if (fabsf(y - e->oy) <= PADDLE_HALF) g[r][COLS - 1] = '#';
    }
    int bc = (int)lroundf(clampf(e->bx, 0, 1) * (COLS - 1)), br = (int)lroundf(clampf(e->by, 0, 1) * (ROWS - 1));
    g[br][bc] = 'O';
    printf("\x1b[H  AI(왼쪽) %d  :  %d  YOU(오른쪽)   난이도 %d   (먼저 %d점)\x1b[K\n +", sa, sh, diff, WIN_SCORE);
    for (int i = 0; i < COLS; i++) putchar('-');
    printf("+\n");
    for (int r = 0; r < ROWS; r++) printf(" |%s|\n", g[r]);
    printf(" +");
    for (int i = 0; i < COLS; i++) putchar('-');
    printf("+\n  w/↑ 위  s/↓ 아래 (누르는 동안 이동)  스페이스 정지  q 종료\x1b[K\n  %s\x1b[K\n", msg);
    fflush(stdout);
}

static int load_net(Net *n, const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "모델을 못 열었습니다: %s\n", p); return 0; }
    int ok = fread(n, sizeof *n, 1, f) == 1;  fclose(f);
    if (!ok) fprintf(stderr, "모델 크기가 맞지 않습니다 (PRED 빌드 필요): %s\n", p);
    return ok;
}

int main(int argc, char **argv)
{
    Net net;
    if (argc >= 2 && !strcmp(argv[1], "selftest")) {      /* 비대화형 검증: AI vs 규칙 추적 상대 */
        if (!load_net(&net, argc > 2 ? argv[2] : "models/final_model.bin")) return 1;
        rs ^= 424242;  for (int i = 0; i < 20; i++) rnd();
        REAL sig[3] = {0.30f, 0.15f, 0.0f};
        for (int d = 0; d < 3; d++) {
            int wins = 0, P = 5000;
            for (int p = 0; p < P; p++) {
                Env e;  env_reset(&e);  int r = 0, st = 0, h;
                while (!r && st++ < 2000) r = step2(&e, ai_act(&net, &e, sig[d]), tracker_act(&e), OPP_SPEED, &h);
                wins += (r == 2);
            }
            printf("난이도 %d (σ=%.2f): AI 승률 %.4f  (규칙 추적 상대, %d점)\n", d + 1, sig[d], (double)wins / P, P);
        }
        return 0;
    }
    int diff = 3;  const char *mp = "models/final_model.bin";
    for (int i = 1; i < argc; i++) { if (argv[i][0] >= '1' && argv[i][0] <= '3' && !argv[i][1]) diff = argv[i][0] - '0'; else mp = argv[i]; }
    if (!load_net(&net, mp)) return 1;
    REAL sigma = diff == 1 ? 0.30f : diff == 2 ? 0.15f : 0.0f;
    REAL hspeed = getenv("HSPEED") ? atof(getenv("HSPEED")) : 0.03f;
    rs ^= (unsigned long long)time(NULL) * 2654435761ULL;  for (int i = 0; i < 20; i++) rnd();

    if (!isatty(0)) { fprintf(stderr, "대화형 터미널에서 실행해 주세요.\n"); return 1; }
    tty_raw();
    int HOLD0 = getenv("HOLD0") ? atoi(getenv("HOLD0")) : 6, HOLD = getenv("HOLD") ? atoi(getenv("HOLD")) : 3;
    int sa = 0, sh = 0, dir = 1, hold = 0, quit = 0, hits_h = 0;   /* hits_h: 사람이 받아친 횟수 (랭킹 점수용) */
    Env e;  env_reset(&e);
    char msg[128] = "준비... 키를 누르는 동안 패들이 움직입니다";
    draw(&e, sa, sh, msg, diff);
    while (!quit && sa < WIN_SCORE && sh < WIN_SCORE) {
        unsigned char c;
        while (read(0, &c, 1) == 1) {
            if (c == 'q') quit = 1;
            else if (c == 'w' || c == 'W') { hold = hold > 0 ? HOLD : HOLD0;  dir = 0; }
            else if (c == 's' || c == 'S') { hold = hold > 0 ? HOLD : HOLD0;  dir = 2; }
            else if (c == ' ') hold = 0;
            else if (c == 27) {                           /* 방향키: ESC [ A/B */
                unsigned char b[2];  usleep(2000);
                if (read(0, &b[0], 1) == 1 && b[0] == '[' && read(0, &b[1], 1) == 1) { if (b[1] == 'A' || b[1] == 'B') { hold = hold > 0 ? HOLD : HOLD0;  dir = b[1] == 'A' ? 0 : 2; } }
            }
        }
        REAL pvx = e.bvx;
        int hit, r = step2(&e, ai_act(&net, &e, sigma), hold > 0 ? dir : 1, hspeed, &hit);
        if (!r && pvx > 0 && e.bvx < 0) hits_h++;        /* 오른쪽으로 가던 공이 되돌아옴 = 사람이 받아침 */
        if (hold > 0) hold--;
        if (r) {
            if (r == 2) { sa++; snprintf(msg, sizeof msg, "AI 득점!"); } else { sh++; snprintf(msg, sizeof msg, "득점! (AI 가 놓쳤습니다)"); }
            draw(&e, sa, sh, msg, diff);  usleep(900000);
            env_reset(&e);  dir = 1;  hold = 0;  msg[0] = 0;
            while (read(0, &c, 1) == 1) ;                 /* 대기 중 쌓인 입력 버림 */
        }
        draw(&e, sa, sh, msg, diff);
        usleep(TICK_US);
    }
    if (!quit) {
        draw(&e, sa, sh, sa > sh ? "AI 승리!" : "당신의 승리!", diff);
        usleep(1200000);
        /* 랭킹: 이름을 일반 입력으로 받아야 하므로 터미널을 원래 모드(블로킹·에코)로 되돌리고 남은 키 입력을 버린다 */
        tty_restore();
        fcntl(0, F_SETFL, fcntl(0, F_GETFL) & ~O_NONBLOCK);
        tcflush(0, TCIFLUSH);
        rank_after_game(RANK_FILE, sh, sa, hits_h, diff);
        printf("\n  다시 하려면 ./play\n");
    }
    return 0;
}
