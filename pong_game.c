/*
 * Pong 게임 (터미널, 순수 C) — 학습된 DQN(v2 모델)이 왼쪽 패들을 조종한다
 * 자료구조 및 실습 기말프로젝트
 *
 *   build : make game        (gcc -O2 -Wall -DPRED -o pong_game pong_game.c ranking.c -lm)
 *   usage : ./pong_game [모델.bin]     기본값 models/final_model.bin
 *
 * 게임 로직(물리·AI·난이도·점수)은 pong_core.h, 랭킹 TOP 10 은 ranking.c 를 그대로 쓴다.
 * 외부 라이브러리 없이 ANSI 이스케이프 + termios 로 그린다.
 *
 * 구성: 1) 터미널 제어  2) 키 입력  3) 출력 버퍼  4) 메인 화면  5) 난이도 / 설정 화면
 *       6) 게임 화면  7) 경기 진행 + 랭킹  8) main
 */
#include <stdarg.h>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#undef IN                       /* windows.h 의 빈 매크로 IN/OUT 이 모델 상수(IN, OUT)와 겹친다 */
#undef OUT
#define STDIN_FILENO  0
#define STDOUT_FILENO 1
#define isatty _isatty
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#endif

#include "pong_core.h"

#define MIN_COLS 64             /* 이보다 작은 터미널에서는 안내문만 출력 */
#define MIN_ROWS 16
#define MAX_FW   90             /* 경기장 최대 크기 (칸) */
#define MAX_FH   30


/* ================= 1. 터미널 제어 ================= */

#ifndef _WIN32
static struct termios term_orig;
static int term_raw = 0;
static volatile sig_atomic_t quit_flag = 0;

/* raw = 1: 줄 단위 입력/에코 끄고 read 가 기다리지 않게(VMIN=0).  raw = 0: 원래 모드 (이름 입력용) */
static void term_mode(int raw)
{
    if (raw) {
        struct termios t = term_orig;
        t.c_lflag &= ~(ICANON | ECHO);
        t.c_cc[VMIN] = 0;  t.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
        fputs("\x1b[?25l\x1b[2J", stdout);                 /* 커서 숨김 */
    } else {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &term_orig);
        fputs("\x1b[?25h\x1b[H\x1b[2J", stdout);           /* 커서 보이기 */
    }
    fflush(stdout);
}
static void term_restore(void)
{
    if (!term_raw) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &term_orig);
    fputs("\x1b[0m\x1b[?25h\x1b[?1049l", stdout);          /* 색 초기화, 커서 보이기, 원래 화면으로 */
    fflush(stdout);
    term_raw = 0;
}
static void on_signal(int sig) { (void)sig;  quit_flag = 1; }
static void term_init(void)
{
    tcgetattr(STDIN_FILENO, &term_orig);
    term_raw = 1;
    atexit(term_restore);
    signal(SIGINT, on_signal);  signal(SIGTERM, on_signal);
    fputs("\x1b[?1049h", stdout);                          /* 대체 화면 */
    term_mode(1);
}
static void term_size(int *cols, int *rows)
{
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) { *cols = w.ws_col;  *rows = w.ws_row; }
    else { *cols = 80;  *rows = 24; }
}
static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void sleep_sec(double d)
{
    if (d <= 0) return;
    struct timespec ts = { (time_t)d, (long)((d - (time_t)d) * 1e9) };
    nanosleep(&ts, NULL);
}

/* 키 입력 기본 함수 (POSIX): 기다리지 않고 읽기 / 최대 sec 초 기다리기 / 쌓인 입력 버리기 */
static int in_getc(unsigned char *c) { return read(STDIN_FILENO, c, 1) == 1; }
static int in_read_n(unsigned char *b, int n) { ssize_t r = read(STDIN_FILENO, b, n);  return r < 0 ? 0 : (int)r; }
static int in_wait(double sec)
{
    fd_set fs;  FD_ZERO(&fs);  FD_SET(STDIN_FILENO, &fs);
    struct timeval tv = { (long)sec, (long)((sec - (long)sec) * 1e6) };
    return select(STDIN_FILENO + 1, &fs, NULL, NULL, &tv) > 0;
}
static void in_flush(void) { tcflush(STDIN_FILENO, TCIFLUSH); }
#else
static int term_raw = 0;
static volatile sig_atomic_t quit_flag = 0;

/* ---- Windows 콘솔: 입력 이벤트를 UTF-8 바이트 열로 바꿔 리눅스 판과 같은 방식으로 처리한다 ---- */
static HANDLE hin, hout;
static DWORD in_mode0, out_mode0;
static UINT cp0;
static unsigned char ibuf[512];
static int ihead, itail;

static void ipush(unsigned char c) { int nx = (itail + 1) % (int)sizeof ibuf;  if (nx != ihead) { ibuf[itail] = c;  itail = nx; } }
static void ipush_utf8(unsigned cp)
{
    if (cp < 0x80) ipush((unsigned char)cp);
    else if (cp < 0x800) { ipush(0xC0 | (cp >> 6));  ipush(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { ipush(0xE0 | (cp >> 12));  ipush(0x80 | ((cp >> 6) & 0x3F));  ipush(0x80 | (cp & 0x3F)); }
    else { ipush(0xF0 | (cp >> 18));  ipush(0x80 | ((cp >> 12) & 0x3F));  ipush(0x80 | ((cp >> 6) & 0x3F));  ipush(0x80 | (cp & 0x3F)); }
}
static void ipump(void)                                    /* 콘솔에 쌓인 키 이벤트를 바이트 버퍼로 옮긴다 */
{
    static unsigned hi = 0;
    INPUT_RECORD r;  DWORD n;
    for (;;) {
        if (!GetNumberOfConsoleInputEvents(hin, &n) || n == 0) break;
        if (!ReadConsoleInputW(hin, &r, 1, &n) || n == 0) break;
        if (r.EventType != KEY_EVENT || !r.Event.KeyEvent.bKeyDown) continue;
        WORD vk = r.Event.KeyEvent.wVirtualKeyCode;
        unsigned ch = r.Event.KeyEvent.uChar.UnicodeChar;
        if (ch == 0) {                                     /* 방향키 -> ESC [ A/B/C/D */
            char c = vk == VK_UP ? 'A' : vk == VK_DOWN ? 'B' : vk == VK_RIGHT ? 'C' : vk == VK_LEFT ? 'D' : 0;
            if (c) { ipush(0x1b);  ipush('[');  ipush((unsigned char)c); }
            continue;
        }
        if (ch >= 0xD800 && ch <= 0xDBFF) { hi = ch;  continue; }          /* 서로게이트 쌍 */
        if (ch >= 0xDC00 && ch <= 0xDFFF) { if (!hi) continue;  ch = 0x10000 + ((hi - 0xD800) << 10) + (ch - 0xDC00); }
        hi = 0;
        ipush_utf8(ch);
    }
}
static int in_getc(unsigned char *c)
{
    ipump();
    if (ihead == itail) return 0;
    *c = ibuf[ihead];  ihead = (ihead + 1) % (int)sizeof ibuf;  return 1;
}
static int in_read_n(unsigned char *b, int n)
{
    int k = 0;
    while (k < n && in_getc(&b[k])) k++;
    return k;
}
static double now(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);  QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
static int in_wait(double sec)
{
    double end = now() + sec;
    for (;;) {
        ipump();
        if (ihead != itail) return 1;
        double left = end - now();
        if (left <= 0) return 0;
        WaitForSingleObject(hin, (DWORD)(left * 1000.0 + 0.5));
    }
}
static void in_flush(void) { FlushConsoleInputBuffer(hin);  ihead = itail = 0; }

static void term_mode(int raw)
{
    if (raw) {
        SetConsoleMode(hin, (in_mode0 & ~(DWORD)(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_QUICK_EDIT_MODE)) | ENABLE_EXTENDED_FLAGS);
        fputs("\x1b[?25l\x1b[2J", stdout);
    } else {
        SetConsoleMode(hin, in_mode0);
        fputs("\x1b[?25h\x1b[H\x1b[2J", stdout);
    }
    fflush(stdout);
}
static void term_restore(void)
{
    if (!term_raw) return;
    fputs("\x1b[0m\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    SetConsoleMode(hin, in_mode0);  SetConsoleMode(hout, out_mode0);  SetConsoleOutputCP(cp0);
    term_raw = 0;
}
static void on_signal(int sig) { (void)sig;  quit_flag = 1; }
static void term_init(void)
{
    hin = GetStdHandle(STD_INPUT_HANDLE);  hout = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleMode(hin, &in_mode0);  GetConsoleMode(hout, &out_mode0);  cp0 = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);                                           /* 한글 출력 */
    SetConsoleMode(hout, out_mode0 | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING);   /* ANSI 색·커서 제어 */
    term_raw = 1;
    atexit(term_restore);
    signal(SIGINT, on_signal);  signal(SIGTERM, on_signal);
    fputs("\x1b[?1049h", stdout);
    term_mode(1);
}
static void term_size(int *cols, int *rows)
{
    CONSOLE_SCREEN_BUFFER_INFO ci;
    if (GetConsoleScreenBufferInfo(hout, &ci)) { *cols = ci.srWindow.Right - ci.srWindow.Left + 1;  *rows = ci.srWindow.Bottom - ci.srWindow.Top + 1; }
    else { *cols = 80;  *rows = 24; }
}
static void sleep_sec(double d)
{
    if (d <= 0) return;
    static HANDLE tm = NULL;                               /* 고해상도 타이머 (Windows 10 1803+), 없으면 Sleep */
    if (!tm) tm = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
    if (tm) {
        LARGE_INTEGER due;  due.QuadPart = -(LONGLONG)(d * 1e7);
        if (SetWaitableTimer(tm, &due, 0, NULL, NULL, FALSE)) { WaitForSingleObject(tm, INFINITE);  return; }
    }
    Sleep((DWORD)(d * 1000.0 + 0.5));
}
#endif


/* ================= 2. 키 입력 ================= */

enum { K_UP = 1000, K_DOWN, K_LEFT, K_RIGHT, K_ENTER, K_ESC };

/* 키 하나를 읽어 정규화해서 돌려준다 (없으면 0).
 * 방향키 / WASD -> K_UP 등, 한글 입력 상태의 ㅈㅁㄴㅇ(WASD)·ㅂ(Q) 도 같은 키로 */
static int read_key(void)
{
    unsigned char c, b[2];
    if (!in_getc(&c)) return 0;
    if (c == 0x1b) {                                         /* ESC [ A/B/C/D */
        int n = in_read_n(b, 2);
        if (n <= 0) return K_ESC;                            /* ESC 단독 */
        if (n < 2 || (b[0] != '[' && b[0] != 'O')) return 0;
        return b[1] == 'A' ? K_UP : b[1] == 'B' ? K_DOWN : b[1] == 'C' ? K_RIGHT : b[1] == 'D' ? K_LEFT : 0;
    }
    if (c == 0xE3) {                                         /* 한글 자모 (UTF-8 3바이트) */
        if (in_read_n(b, 2) != 2) return 0;
        if (b[0] == 0x85 && b[1] == 0x88) return K_UP;       /* ㅈ */
        if (b[0] == 0x84 && b[1] == 0xB4) return K_DOWN;     /* ㄴ */
        if (b[0] == 0x85 && b[1] == 0x81) return K_LEFT;     /* ㅁ */
        if (b[0] == 0x85 && b[1] == 0x87) return K_RIGHT;    /* ㅇ */
        if (b[0] == 0x85 && b[1] == 0x82) return 'q';        /* ㅂ */
        return 0;
    }
    if (c == '\n' || c == '\r') return K_ENTER;
    if (c >= 'A' && c <= 'Z') c += 32;
    return c == 'w' ? K_UP : c == 's' ? K_DOWN : c == 'a' ? K_LEFT : c == 'd' ? K_RIGHT : c;
}

/* 최대 sec 초 동안 키를 기다린다 (메뉴용: 시간이 지나면 0 -> 다시 그리기, 창 크기 변경 반영) */
static int wait_key(double sec)
{
    if (!in_wait(sec)) return 0;
    return read_key();
}


/* ================= 3. 출력 버퍼 ================= */

/* 한 화면을 버퍼에 다 모았다가 write 한 번으로 내보낸다 (깜빡임 방지) */
static char out[1 << 16];
static size_t outn;

static void ap(const char *s)
{
    size_t n = strlen(s);
    if (outn + n < sizeof out) { memcpy(out + outn, s, n);  outn += n; }
}
static void apf(const char *fmt, ...)
{
    char tmp[512];
    va_list va;  va_start(va, fmt);  vsnprintf(tmp, sizeof tmp, fmt, va);  va_end(va);
    ap(tmp);
}
static void pad(int n) { while (n-- > 0) ap(" "); }
static void eol(void) { ap("\x1b[K\n"); }
static void flush_out(void)
{
    ap("\x1b[J");                                          /* 남은 아래쪽 지우기 */
    for (size_t i = 0; i < outn; ) {
        long n = (long)write(STDOUT_FILENO, out + i, (unsigned)(outn - i));
        if (n <= 0) break;
        i += n;
    }
    outn = 0;
}
/* 가운데 정렬 한 줄. style = ANSI 스타일 ("" 가능) */
static void cline(int cols, const char *style, const char *text)
{
    pad((cols - text_width(text)) / 2);
    apf("%s%s\x1b[0m", style, text);
    eol();
}
/* 화면이 너무 작을 때 */
static int too_small(int cols, int rows)
{
    if (cols >= MIN_COLS && rows >= MIN_ROWS) return 0;
    ap("\x1b[H");
    apf("터미널 창이 너무 작습니다 (%dx%d). 최소 %dx%d 로 키워주세요.", cols, rows, MIN_COLS, MIN_ROWS);
    eol();  flush_out();
    return 1;
}


/* ================= 4. 메인 화면 ================= */

static const char *TITLE[] = {             /* "PONG" 큰 글씨 */
    "██████╗  ██████╗ ███╗   ██╗ ██████╗ ",
    "██╔══██╗██╔═══██╗████╗  ██║██╔════╝ ",
    "██████╔╝██║   ██║██╔██╗ ██║██║  ███╗",
    "██╔═══╝ ██║   ██║██║╚██╗██║██║   ██║",
    "██║     ╚██████╔╝██║ ╚████║╚██████╔╝",
    "╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ",
};
#define TITLE_LINES 6
static const int TITLE_COLOR[TITLE_LINES] = { 51, 45, 39, 33, 27, 21 };   /* 256색: 하늘색 -> 파랑 */

/* big = 1: 6줄짜리 큰 글씨, 0: 한 줄 */
static void draw_title(int cols, int big)
{
    if (!big) { cline(cols, "\x1b[1;96m", "P  O  N  G");  return; }
    for (int i = 0; i < TITLE_LINES; i++) {
        char st[32];  snprintf(st, sizeof st, "\x1b[1;38;5;%dm", TITLE_COLOR[i]);
        cline(cols, st, TITLE[i]);
    }
}

/* 메뉴 항목 한 줄 (선택된 항목은 ▶ + 반전). 블록 왼쪽 위치 left 에 맞춰 출력 */
static void menu_item(int left, int selected, const char *text, const char *extra)
{
    pad(left);
    if (selected) apf("\x1b[1;96m▶ \x1b[7m %s \x1b[0m", text);
    else          apf("   %s ", text);
    if (extra) apf("  \x1b[2m%s\x1b[0m", extra);
    eol();
}

static const char *MAIN_ITEMS[4] = { "1. 시작", "2. 난이도", "3. 설정", "4. 종료" };

/* 메인 화면. 고른 항목 0~3 반환 (종료 신호면 3) */
static int main_menu(int *sel, const Settings *st)
{
    Ranking rk;  rank_load(&rk, RANK_FILE);
    for (;;) {
        int cols, rows;  term_size(&cols, &rows);
        if (!too_small(cols, rows)) {
            /* 랭킹 아래 줄 수: 제목·메뉴(큰 제목 15줄 / 작은 제목 10줄) + 랭킹 머리·빈 줄·안내·여유 4줄.
             * 랭킹 TOP 10 이 다 들어가는 게 우선 -> 자리가 모자라면 제목을 한 줄로 줄인다 */
            int need = rk.n > 0 ? rk.n : 1;
            int big  = rows - 19 >= need;
            int show = rows - (big ? 19 : 14);
            if (show > RANK_MAX) show = RANK_MAX;

            ap("\x1b[H");  eol();
            draw_title(cols, big);
            eol();
            cline(cols, "\x1b[2m", "학습된 DQN 신경망과 대결하는 터미널 Pong");
            eol();
            int left = (cols - 30) / 2;
            char diff[48];  snprintf(diff, sizeof diff, "[%s]", DIFFS[st->diff - 1].name);
            for (int i = 0; i < 4; i++) menu_item(left, *sel == i, MAIN_ITEMS[i], i == 1 ? diff : NULL);
            eol();

            /* 랭킹 TOP 10 (창이 작으면 들어가는 만큼만 + 안내) */
            if (show > 0) {
                char head[96];
                if (show < rk.n) snprintf(head, sizeof head, "── 랭킹 TOP %d (1~%d위 · 창을 키우면 전체) ──", RANK_MAX, show);
                else             snprintf(head, sizeof head, "── 랭킹 TOP %d ──", RANK_MAX);
                cline(cols, "\x1b[1;33m", head);
                if (rk.n == 0) cline(cols, "\x1b[2m", "(아직 기록이 없습니다)");
                for (int i = 0; i < rk.n && i < show; i++) {
                    const RankEntry *x = &rk.e[i];
                    pad((cols - 34) / 2);
                    apf("\x1b[%sm%2d  %s", i == 0 ? "1;33" : "0", i + 1, x->name);
                    pad(16 - text_width(x->name));
                    apf("%6d점  Lv%d\x1b[0m", x->score, x->diff);
                    eol();
                }
                eol();
            }
            cline(cols, "\x1b[2m", "↑↓ 이동 · Enter 선택 · 숫자키 바로 선택");
            flush_out();
        }

        int k = wait_key(0.25);
        if (quit_flag) return 3;
        if (k >= '1' && k <= '4') { *sel = k - '1';  return *sel; }
        if (k == K_UP)   *sel = (*sel + 3) % 4;
        if (k == K_DOWN) *sel = (*sel + 1) % 4;
        if (k == K_ENTER || k == ' ') return *sel;
        if (k == 'q' || k == K_ESC) { *sel = 3;  return 3; }
    }
}


/* ================= 5. 난이도 / 설정 화면 ================= */

static void small_header(int cols, const char *title)
{
    ap("\x1b[H");  eol();
    cline(cols, "\x1b[1;96m", "P  O  N  G");
    eol();
    cline(cols, "\x1b[1m", title);
    eol();
}

static void difficulty_menu(Settings *st)
{
    int sel = st->diff - 1;
    for (;;) {
        int cols, rows;  term_size(&cols, &rows);
        if (!too_small(cols, rows)) {
            small_header(cols, "난이도 선택");
            int left = (cols - 56) / 2;  if (left < 1) left = 1;
            for (int i = 0; i < N_DIFFS; i++) {
                const Difficulty *d = &DIFFS[i];
                char name[48];  snprintf(name, sizeof name, "%d. %s", i + 1, d->name);
                menu_item(left, sel == i, name, st->diff == i + 1 ? "(현재)" : NULL);
                int m10 = rank_calc_score(0, 0, 1, i + 1) * 10 / SCORE_WIN_BONUS;    /* 랭킹 배율 ×10 */
                pad(left + 6);  apf("\x1b[2m%s", d->desc);  eol();
                pad(left + 6);  apf("\x1b[2m규칙 봇 상대 AI 승률 약 %d%% · 랭킹 점수 ×%d.%d\x1b[0m", d->tracker_win, m10 / 10, m10 % 10);
                eol();  eol();
            }
            cline(cols, "\x1b[2m", "↑↓ 이동 · Enter 선택 · Esc 뒤로");
            flush_out();
        }
        int k = wait_key(0.25);
        if (quit_flag || k == 'q' || k == K_ESC) return;
        if (k >= '1' && k < '1' + N_DIFFS) { st->diff = k - '0';  return; }
        if (k == K_UP)   sel = (sel + N_DIFFS - 1) % N_DIFFS;
        if (k == K_DOWN) sel = (sel + 1) % N_DIFFS;
        if (k == K_ENTER || k == ' ') { st->diff = sel + 1;  return; }
    }
}

static void settings_menu(Settings *st)
{
    enum { S_WIN, S_SPEED, S_Q, S_BACK, N_S };
    int sel = 0;
    for (;;) {
        int cols, rows;  term_size(&cols, &rows);
        if (!too_small(cols, rows)) {
            small_header(cols, "설정");
            int left = (cols - 44) / 2;
            char v[N_S][48];
            snprintf(v[S_WIN],   48, "◀ %d점 선승 ▶", WIN_OPTS[st->win_i]);
            snprintf(v[S_SPEED], 48, "◀ %s (%dfps) ▶", SPEED_OPTS[st->speed_i].name, SPEED_OPTS[st->speed_i].fps);
            snprintf(v[S_Q],     48, "◀ %s ▶", st->show_q ? "켬" : "끔");
            const char *label[N_S] = { "목표 점수     ", "게임 속도     ", "AI 판단 표시  ", "돌아가기" };
            for (int i = 0; i < N_S; i++) {
                pad(left);
                if (sel == i) apf("\x1b[1;96m▶ \x1b[7m %s \x1b[0m", label[i]);
                else          apf("   %s ", label[i]);
                if (i != S_BACK) apf(sel == i ? "  \x1b[1m%s\x1b[0m" : "  %s", v[i]);
                eol();  eol();
            }
            cline(cols, "\x1b[2m", "AI 판단 표시 = 화면 아래에 AI 가 본 Q값(위/정지/아래)");
            cline(cols, "\x1b[2m", "게임 속도는 실제 시간 기준 속도 (물리·AI 는 그대로)");
            eol();
            cline(cols, "\x1b[2m", "↑↓ 항목 · ←→ 값 바꾸기 · Esc 뒤로");
            flush_out();
        }
        int k = wait_key(0.25);
        if (quit_flag || k == 'q' || k == K_ESC) return;
        if (k == K_UP)   sel = (sel + N_S - 1) % N_S;
        if (k == K_DOWN) sel = (sel + 1) % N_S;
        int dv = k == K_RIGHT || k == K_ENTER || k == ' ' ? 1 : k == K_LEFT ? -1 : 0;
        if (!dv) continue;
        if (sel == S_BACK) { if (k == K_ENTER || k == ' ') return;  continue; }
        if (sel == S_WIN)   st->win_i   = (st->win_i + dv + N_WIN_OPTS) % N_WIN_OPTS;
        if (sel == S_SPEED) st->speed_i = (st->speed_i + dv + N_SPEED_OPTS) % N_SPEED_OPTS;
        if (sel == S_Q)     st->show_q  = !st->show_q;
    }
}


/* ================= 6. 게임 화면 ================= */

enum { PX_NONE, PX_AI, PX_ME, PX_BALL };
static const int PX_COLOR[] = { 0, 96, 93, 97 };           /* AI 하늘색, 사람 노랑, 공 흰색 */

/* 셀 하나 = 위/아래 픽셀 2개. 반쪽 블록(▀ ▄)에 글자색/배경색을 써서 세로 해상도를 2배로 */
static void put_cell(int top, int bot, int net)
{
    if (!top && !bot)    ap(net ? "\x1b[90m│\x1b[0m" : " ");
    else if (top && bot) apf("\x1b[%d;%dm▀\x1b[0m", PX_COLOR[top], PX_COLOR[bot] + 10);
    else if (top)        apf("\x1b[%dm▀\x1b[0m", PX_COLOR[top]);
    else                 apf("\x1b[%dm▄\x1b[0m", PX_COLOR[bot]);
}

static void render(const Game *g, const Settings *st, int paused)
{
    int cols, rows;  term_size(&cols, &rows);
    if (too_small(cols, rows)) return;
    ap("\x1b[H");

    /* 경기장 fw x fh 칸 (오른쪽 끝 한 칸은 비워 둔다: 마지막 열에서의 줄바꿈 문제 회피) */
    int fw = cols - 3 < MAX_FW ? cols - 3 : MAX_FW;
    int fh = rows - (st->show_q ? 5 : 4);  if (fh > MAX_FH) fh = MAX_FH;
    int ph = fh * 2;                                       /* 세로 픽셀 수 */

    /* 점수 줄 */
    char mid[32];  snprintf(mid, sizeof mid, "   %d : %d   ", g->score_ai, g->score_me);
    char info[64]; snprintf(info, sizeof info, "  [%s · %d점 선승]", DIFFS[g->diff - 1].name, g->win_score);
    pad((fw + 2 - text_width("DQN AI") - text_width(mid) - text_width("YOU")) / 2);
    apf("\x1b[1;%dmDQN AI\x1b[0;1m%s\x1b[%dmYOU\x1b[0m\x1b[2m%s\x1b[0m", PX_COLOR[PX_AI], mid, PX_COLOR[PX_ME], info);
    eol();

    /* 경기장: 좌표 0~1 을 칸/픽셀로 변환. 공은 패들 사이 1 ~ fw-2 칸에 */
    const Env *e = &g->e;
    int bc = 1 + (int)floorf(e->bx * (fw - 2));
    int bp = (int)(e->by * ph);
    bc = bc < 0 ? 0 : bc > fw - 1 ? fw - 1 : bc;
    bp = bp < 0 ? 0 : bp > ph - 1 ? ph - 1 : bp;

    const char *msg = paused ? "일시정지" : g->msg[0] ? g->msg : NULL;
    int mw = msg ? text_width(msg) + 2 : 0, mc = (fw - mw) / 2, mr = fh / 2;

    ap("┌");  for (int c = 0; c < fw; c++) ap("─");  ap("┐");  eol();
    for (int r = 0; r < fh; r++) {
        ap("│");
        for (int c = 0; c < fw; c++) {
            if (msg && r == mr && c == mc) { apf("\x1b[1;7m %s \x1b[0m", msg);  c += mw - 1;  continue; }
            int px[2];
            for (int k = 0; k < 2; k++) {
                int p = 2 * r + k;
                REAL y = (p + 0.5f) / ph;                  /* 이 픽셀 중심의 y 좌표 */
                px[k] = (c == bc && p == bp)                             ? PX_BALL
                      : (c == 0      && fabsf(y - e->py) <= PADDLE_HALF) ? PX_AI
                      : (c == fw - 1 && fabsf(y - e->oy) <= PADDLE_HALF) ? PX_ME
                      : PX_NONE;
            }
            put_cell(px[0], px[1], c == fw / 2 && r % 2 == 0);
        }
        ap("│");  eol();
    }
    ap("└");  for (int c = 0; c < fw; c++) ap("─");  ap("┘");  eol();

    /* AI 가 본 Q 값과 고른 행동 (행동 간 차이가 작아서 소수 셋째 자리까지) */
    if (st->show_q) {
        static const char *ACT[OUT] = { "▲ 위", "■ 정지", "▼ 아래" };
        apf(" \x1b[%dmAI 판단\x1b[0m ", PX_COLOR[PX_AI]);
        for (int i = 0; i < OUT; i++)
            apf(i == g->ai_act ? " \x1b[1;7m %s %+.3f \x1b[0m" : "  %s %+.3f ", ACT[i], (double)g->q[i]);
        eol();
    }

    /* 받아친 횟수 + 조작법 (마지막 줄은 줄바꿈 없이: 화면이 밀려 올라가지 않게) */
    apf(" \x1b[%dm받아침 %d\x1b[0m \x1b[2m·  ↑↓ / W S 이동   Space 일시정지   Q 메뉴로(기록 안 됨)\x1b[0m\x1b[K",
        PX_COLOR[PX_ME], g->hits_me);
    flush_out();
}


/* ================= 7. 경기 결과 + 랭킹 ================= */

/* 경기 종료 화면에 필요한 값 (점수 계산·순위 판정·저장은 ranking.c 함수 그대로) */
typedef struct {
    const Game *g;
    int  won, base, m10, score;     /* base = 배율 전 점수, m10 = 난이도 배율 ×10 */
    Ranking rk;
    int  qualifies, expect, pos;    /* expect = 들어갈 순위 인덱스, pos = 등록된 인덱스 (-1 없음) */
    char name[RANK_NAME_LEN];       /* 입력 중인 이름 (UTF-8) */
    int  full;                      /* 길이 제한에 걸려 마지막 글자를 못 넣었음 */
} Result;

/* 결과 요약 5줄: 승패 / 스코어 / 빈 줄 / 점수 계산식 / 최종 점수 */
static void draw_summary(int cols, const Result *R)
{
    const Game *g = R->g;
    char t[160];
    cline(cols, R->won ? "\x1b[1;93m" : "\x1b[1;91m", R->won ? "★  승리!  ★" : "패배");
    snprintf(t, sizeof t, "YOU %d : %d DQN AI  ·  %s", g->score_me, g->score_ai, DIFFS[g->diff - 1].name);
    cline(cols, "\x1b[1m", t);
    eol();
    snprintf(t, sizeof t, "득점 %d×%d + 받아침 %d×%d + 승리 %d = %d   × 배율 %d.%d",
             g->score_me, SCORE_PER_POINT, g->hits_me, SCORE_PER_HIT, R->won ? SCORE_WIN_BONUS : 0,
             R->base, R->m10 / 10, R->m10 % 10);
    cline(cols, "\x1b[2m", t);
    snprintf(t, sizeof t, "이번 점수  %d점", R->score);
    cline(cols, "\x1b[1;96m", t);
}

#define BOX_W 34                    /* 이름 입력 상자 안쪽 폭 (칸) */

/* 이름 입력 화면을 그리고 입력칸에 커서를 둔다 (한글 조합 중인 글자도 그 자리에 보이도록) */
static void draw_name_input(const Result *R)
{
    int cols, rows;  term_size(&cols, &rows);
    if (too_small(cols, rows)) return;
    const int lines = 13;                                  /* 이 화면의 줄 수 */
    int top = (rows - lines) / 2;  if (top < 0) top = 0;
    int left = (cols - (BOX_W + 2)) / 2;

    ap("\x1b[?25l\x1b[H");
    for (int i = 0; i < top; i++) eol();
    draw_summary(cols, R);
    eol();
    char t[96];
    snprintf(t, sizeof t, "TOP %d 진입!  예상 %d위", RANK_MAX, R->expect + 1);
    cline(cols, "\x1b[1;93m", t);
    eol();

    const char *label = "  이름  ▸ ";
    int lw = text_width(label), nw = text_width(R->name);
    pad(left);  ap("\x1b[93m┌");  for (int i = 0; i < BOX_W; i++) ap("─");  ap("┐\x1b[0m");  eol();
    pad(left);  apf("\x1b[93m│\x1b[0;1m%s\x1b[0;1;97m%s\x1b[0m", label, R->name);
    pad(BOX_W - lw - nw);  ap("\x1b[93m│\x1b[0m");  eol();
    pad(left);  ap("\x1b[93m└");  for (int i = 0; i < BOX_W; i++) ap("─");  ap("┘\x1b[0m");  eol();

    int rest = RANK_NAME_LEN - 1 - (int)strlen(R->name);
    snprintf(t, sizeof t, R->full ? "더 이상 넣을 수 없습니다 (최대 한글 5자 / 영문 15자)"
                                  : "남은 길이: 한글 %d자 / 영문 %d자", rest / 3, rest);
    cline(cols, R->full ? "\x1b[91m" : "\x1b[2m", t);
    cline(cols, "\x1b[2m", "Enter 등록 · Backspace 지우기 · 빈 이름은 PLAYER");
    flush_out();

    /* 커서를 입력칸 글자 끝으로 옮기고 보이게 (flush_out 뒤에 따로 써야 \x1b[J 에 지워지지 않음) */
    char cur[48];
    int n = snprintf(cur, sizeof cur, "\x1b[%d;%dH\x1b[?25h", top + 10, left + 1 + lw + nw + 1);   /* 1부터 세는 열: 여백 + │ + 라벨 + 이름 다음 칸 */
    if (write(STDOUT_FILENO, cur, n) < 0) { /* 화면 출력 실패는 무시 */ }
}

/* 마지막 UTF-8 글자 하나 지우기 */
static void name_backspace(char *s)
{
    size_t k = strlen(s);
    while (k > 0 && ((unsigned char)s[k - 1] & 0xC0) == 0x80) k--;   /* 이어지는 바이트 건너뛰기 */
    if (k > 0) k--;
    s[k] = 0;
}

/* raw 모드에서 이름을 직접 입력받는다. Enter 면 1, Ctrl+C 면 0 */
static int name_input(Result *R)
{
    for (;;) {
        draw_name_input(R);
        int sel = in_wait(0.25);                           /* 시간이 지나면 다시 그리기 (창 크기 변경 반영) */
        if (quit_flag) return 0;
        if (!sel) continue;

        unsigned char c, b[8];
        if (!in_getc(&c)) continue;
        size_t len = strlen(R->name);
        R->full = 0;

        if (c == '\r' || c == '\n') return 1;
        if (c == 0x7f || c == 0x08) { name_backspace(R->name);  continue; }
        if (c == 0x15) { R->name[0] = 0;  continue; }      /* Ctrl+U: 전부 지우기 */
        if (c == 0x1b) { in_read_n(b, sizeof b);  continue; }   /* 방향키 등은 버림 */

        int n = c >= 0x20 && c < 0x7f ? 1 : c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3
              : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
        if (!n) continue;                                  /* 그 밖의 제어 문자 무시 */
        b[0] = c == ' ' ? '_' : c;                         /* 파일 형식상 공백 불가 -> 보이는 그대로 '_' */
        int ok = 1;
        for (int i = 1; i < n; i++) {                      /* 나머지 바이트 (보통 이미 와 있음) */
            if (!in_wait(0.05) || !in_getc(&b[i])
                || (b[i] & 0xC0) != 0x80) { ok = 0;  break; }
        }
        if (!ok) continue;
        if (len + n > RANK_NAME_LEN - 1) { R->full = 1;  continue; }
        memcpy(R->name + len, b, n);
        R->name[len + n] = 0;
    }
}

/* 결과 + 랭킹 TOP 10 표 (방금 등록한 기록 강조) */
static void draw_board(const Result *R)
{
    int cols, rows;  term_size(&cols, &rows);
    if (too_small(cols, rows)) return;
    int n = R->rk.n;
    int show = n < 1 ? 1 : n;
    if (show > rows - 11) show = rows - 11;                /* 창이 작으면 들어가는 만큼 */
    int start = R->pos >= show ? R->pos - show + 1 : 0;   /* 방금 기록은 꼭 보이게 */
    int lines = 10 + show;
    int top = (rows - lines) / 2;  if (top < 0) top = 0;

    ap("\x1b[?25l\x1b[H");
    for (int i = 0; i < top; i++) eol();
    draw_summary(cols, R);
    char t[128];
    if (R->pos >= 0)          snprintf(t, sizeof t, "%s 님, %d위로 등록되었습니다!", R->rk.e[R->pos].name, R->pos + 1);
    else if (!R->qualifies)   snprintf(t, sizeof t, "아쉽게도 TOP %d 밖입니다 (10위 %d점)", RANK_MAX, R->rk.e[RANK_MAX - 1].score);
    else                      snprintf(t, sizeof t, " ");
    cline(cols, R->pos >= 0 ? "\x1b[1;93m" : "\x1b[2m", t);
    eol();

    /* 표: 순위(3) 3칸 이름(16) 2칸 점수(6)+점 3칸 Lv 3칸 스코어 */
    const int TW = 48;
    int left = (cols - TW) / 2;
    snprintf(t, sizeof t, "── 랭킹 TOP %d ──", RANK_MAX);
    cline(cols, "\x1b[1;33m", t);
    pad(left);  ap("\x1b[2m순위  이름                  점수  난이도  스코어\x1b[0m");  eol();
    if (n == 0) cline(cols, "\x1b[2m", "(아직 기록이 없습니다)");
    for (int i = start; i < n && i < start + show; i++) {
        const RankEntry *x = &R->rk.e[i];
        pad(left);
        apf("%s%3d   %s", i == R->pos ? "\x1b[1;93m" : i == 0 ? "\x1b[33m" : "", i + 1, x->name);
        pad(16 - text_width(x->name));
        apf("  %6d점   Lv%d   %2d : %-2d%s\x1b[0m", x->score, x->diff, x->my_pts, x->ai_pts, i == R->pos ? "  ◀ NEW" : "");
        eol();
    }
    eol();
    cline(cols, "\x1b[2m", "Enter 를 누르면 메인 화면으로");
    flush_out();
}

/* 경기 종료 후: 점수 계산 → TOP 10 이면 이름 입력 → ranking.txt 저장 → 랭킹 표 */
static void ranking_screen(const Game *g)
{
    Result R;
    memset(&R, 0, sizeof R);
    R.g = g;
    R.won = g->score_me > g->score_ai;
    R.base = g->score_me * SCORE_PER_POINT + g->hits_me * SCORE_PER_HIT + (R.won ? SCORE_WIN_BONUS : 0);
    R.m10 = rank_calc_score(0, 0, 1, g->diff) * 10 / SCORE_WIN_BONUS;   /* 난이도 배율 ×10 */
    R.score = game_rank_score(g);
    R.pos = -1;
    rank_load(&R.rk, RANK_FILE);
    R.qualifies = rank_qualifies(&R.rk, R.score);
    while (R.expect < R.rk.n && R.rk.e[R.expect].score >= R.score) R.expect++;   /* 동점이면 기존 기록이 위 */

    in_flush();                                            /* 경기 중 쌓인 키 입력 버림 */
    fputs("\x1b[2J", stdout);  fflush(stdout);

    if (R.qualifies) {
        if (!name_input(&R)) { fputs("\x1b[?25l", stdout);  fflush(stdout);  return; }   /* Ctrl+C: 저장 안 함 */
        RankEntry x;
        rank_set_name(&x, R.name);
        x.score = R.score;  x.diff = g->diff;  x.my_pts = g->score_me;  x.ai_pts = g->score_ai;
        R.pos = rank_insert(&R.rk, &x);
        if (!rank_save(&R.rk, RANK_FILE)) R.pos = -1;      /* 저장 실패면 등록 안 된 것으로 표시 */
        fputs("\x1b[?25l\x1b[2J", stdout);  fflush(stdout);
    }
    for (;;) {
        draw_board(&R);
        int k = wait_key(0.25);
        if (quit_flag || k == K_ENTER || k == ' ' || k == 'q' || k == K_ESC) break;
    }
}

/* 한 경기. Q 를 누르면 기록 없이 메뉴로 */
static void play(const Settings *st)
{
    int fps = SPEED_OPTS[st->speed_i].fps;
    /* 터미널은 키를 뗀 순간을 알 수 없다 -> '키 반복 입력이 끊기면 멈춤'.
     * 첫 입력은 키 반복이 시작되기 전 지연을 메우려고 조금 더 길게 유지 */
    int hold_first = fps / 5, hold_repeat = fps / 8;
    Game g;  game_start(&g, st->diff, WIN_OPTS[st->win_i], fps);
    int key_dir = 0, key_frames = 0, paused = 0;
    double next = now();

    while (!quit_flag) {
        for (int k; (k = read_key()); ) {
            if (k == 'q' || k == K_ESC) return;
            if (k == ' ' || k == 'p') paused = !paused;
            else if ((k == K_UP || k == K_DOWN) && !paused) {
                int d = k == K_UP ? -1 : 1;
                key_frames = (key_frames > 0 && key_dir == d) ? hold_repeat : hold_first;
                key_dir = d;
            }
        }
        int dir = 0;
        if (!paused && key_frames > 0) { dir = key_dir;  key_frames--; }
        int ev = paused ? EV_NONE : game_frame(&g, dir, fps);
        if (ev != EV_NONE) key_frames = 0;
        render(&g, st, paused);

        if (ev == EV_GAME_OVER) {
            sleep_sec(1.5);
            ranking_screen(&g);
            return;
        }
        next += 1.0 / fps;                                 /* 고정 프레임레이트 */
        double d = next - now();
        if (d > 0) sleep_sec(d);
        else next = now();                                 /* 밀렸으면 따라잡지 않고 다시 맞춘다 */
    }
}


/* ================= 8. main ================= */

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : MODEL_PATH_DEFAULT;
    char err[256];
    if (load_model(path, err, sizeof err)) { fprintf(stderr, "%s\n", err);  return 1; }
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        fprintf(stderr, "터미널에서 실행해 주세요.\n");
        return 1;
    }
    core_seed();

    term_init();
    Settings st = settings_default();
    int sel = 0;
    while (!quit_flag) {
        int c = main_menu(&sel, &st);
        if (c == 0)      play(&st);
        else if (c == 1) difficulty_menu(&st);
        else if (c == 2) settings_menu(&st);
        else break;
    }
    term_restore();
    return 0;
}
