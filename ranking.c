/* ranking.c — 랭킹 TOP 10 · 점수 기록 구현 (담당: 서태호). 설명은 ranking.h 참고. */
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS      /* Visual Studio 의 fopen/sscanf 경고 끄기 */
#endif
#include "ranking.h"
#include <stdio.h>
#include <string.h>

static const int DIFF_MULT_X10[4] = { 10, 10, 15, 20 };   /* [난이도] → 배율×10 (0번은 안전용) */

static int clamp_diff(int d) { return d < 1 ? 1 : d > 3 ? 3 : d; }

int rank_calc_score(int my_pts, int hits, int won, int diff)
{
    int base = my_pts * SCORE_PER_POINT + hits * SCORE_PER_HIT + (won ? SCORE_WIN_BONUS : 0);
    return base * DIFF_MULT_X10[clamp_diff(diff)] / 10;
}

/* 삽입 정렬의 한 단계: a[i] 를 점수가 더 높은 기록 바로 뒤까지 앞으로 밀어 올린다.
 * '<' 비교라 동점은 넘어가지 않음 → 먼저 세운 기록이 위 (안정 정렬). 최종 인덱스 반환. */
static int sift_up(RankEntry *a, int i)
{
    RankEntry key = a[i];
    while (i > 0 && a[i - 1].score < key.score) {
        a[i] = a[i - 1];
        i--;
    }
    a[i] = key;
    return i;
}

int rank_qualifies(const Ranking *r, int score)
{
    return r->n < RANK_MAX || score > r->e[RANK_MAX - 1].score;
}

int rank_insert(Ranking *r, const RankEntry *x)
{
    int i;
    if (r->n < RANK_MAX)                       i = r->n++;        /* 빈 자리: 맨 뒤에 추가 */
    else if (x->score > r->e[RANK_MAX - 1].score) i = RANK_MAX - 1;  /* 꽉 참: 10등을 밀어냄 */
    else                                        return -1;          /* 순위 밖 */
    r->e[i] = *x;
    return sift_up(r->e, i);
}

void rank_set_name(RankEntry *x, const char *raw)
{
    size_t k = 0;
    while (*raw == ' ' || *raw == '\t') raw++;                   /* 앞 공백 제거 */
    for (; raw[k] && raw[k] != '\n' && raw[k] != '\r' && k < RANK_NAME_LEN - 1; k++)
        x->name[k] = (raw[k] == ' ' || raw[k] == '\t') ? '_' : raw[k];   /* 파일 형식상 공백 불가 */
    /* 길이 제한으로 잘렸으면 한글(UTF-8 여러 바이트) 글자가 중간에 끊기지 않게 글자 경계까지 되돌림 */
    if (raw[k] && raw[k] != '\n' && raw[k] != '\r')
        while (k > 0 && ((unsigned char)raw[k] & 0xC0) == 0x80) k--;
    while (k > 0 && x->name[k - 1] == '_') k--;                  /* 뒤 공백(→'_') 제거 */
    x->name[k] = '\0';
    if (k == 0) strcpy(x->name, "PLAYER");
}

void rank_load(Ranking *r, const char *path)
{
    char line[128];
    FILE *f = fopen(path, "r");
    r->n = 0;
    if (!f) return;                                              /* 첫 실행: 빈 랭킹 */
    while (fgets(line, sizeof line, f)) {
        RankEntry x;
        if (sscanf(line, "%15s %d %d %d %d", x.name, &x.score, &x.diff, &x.my_pts, &x.ai_pts) == 5)
            rank_insert(r, &x);              /* 파일이 손으로 고쳐져 순서가 섞여도 정렬되어 들어감 */
    }
    fclose(f);
}

int rank_save(const Ranking *r, const char *path)
{
    int i;
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    for (i = 0; i < r->n; i++)
        fprintf(f, "%s %d %d %d %d\n", r->e[i].name, r->e[i].score, r->e[i].diff, r->e[i].my_pts, r->e[i].ai_pts);
    return fclose(f) == 0;
}

/* 터미널에 보이는 폭 (한글 등 여러 바이트 글자는 2칸으로 계산) — 표 줄맞춤용 */
static int display_width(const char *s)
{
    int w = 0;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c < 0x80) w += 1;
        else if ((c & 0xC0) == 0xC0) w += 2;                     /* 여러 바이트 글자의 첫 바이트만 셈 */
    }
    return w;
}

void rank_print(const Ranking *r, int highlight)
{
    int i, pad;
    printf("\n  ============ RANKING TOP %d ============\n", RANK_MAX);
    printf("  순위  이름              점수  난이도  스코어\n");
    printf("  ----  ----------------  ----  ------  ------\n");
    if (r->n == 0) printf("  (아직 기록이 없습니다)\n");
    for (i = 0; i < r->n; i++) {
        const RankEntry *x = &r->e[i];
        if (i == highlight) printf("\x1b[1;33m");                /* 방금 기록: 굵은 노란색 */
        printf("  %3d   %s", i + 1, x->name);
        for (pad = 16 - display_width(x->name); pad > 0; pad--) putchar(' ');
        printf("  %4d   Lv%d    %2d:%-2d", x->score, x->diff, x->my_pts, x->ai_pts);
        if (i == highlight) printf("  ◀ NEW\x1b[0m");
        putchar('\n');
    }
    printf("  =======================================\n");
    fflush(stdout);
}

void rank_after_game(const char *path, int my_pts, int ai_pts, int hits, int diff)
{
    Ranking r;
    RankEntry x;
    char buf[128];
    int won = my_pts > ai_pts, pos = -1;

    diff = clamp_diff(diff);
    x.score  = rank_calc_score(my_pts, hits, won, diff);
    x.diff   = diff;
    x.my_pts = my_pts;
    x.ai_pts = ai_pts;

    printf("\n  [%s]  %d : %d  (난이도 %d)\n", won ? "승리" : "패배", my_pts, ai_pts, diff);
    printf("  점수 = (득점 %d×%d + 받아침 %d×%d + 승리 %d) × %d.%d = %d점\n",
           my_pts, SCORE_PER_POINT, hits, SCORE_PER_HIT, won ? SCORE_WIN_BONUS : 0,
           DIFF_MULT_X10[diff] / 10, DIFF_MULT_X10[diff] % 10, x.score);

    rank_load(&r, path);
    if (rank_qualifies(&r, x.score)) {
        printf("  TOP %d 진입! 이름을 입력하세요 (최대 15바이트): ", RANK_MAX);
        fflush(stdout);
        rank_set_name(&x, fgets(buf, sizeof buf, stdin) ? buf : "");
        pos = rank_insert(&r, &x);
        if (!rank_save(&r, path)) printf("  (경고: %s 에 저장하지 못했습니다)\n", path);
    } else {
        printf("  아쉽게도 TOP %d 밖입니다. (10등: %d점)\n", RANK_MAX, r.e[RANK_MAX - 1].score);
    }
    rank_print(&r, pos);
}
