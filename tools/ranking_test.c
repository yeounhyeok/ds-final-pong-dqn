/* ranking_test.c — 랭킹 모듈 자동 테스트.  빌드·실행: make ranking-test */
#include "../ranking.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static RankEntry mk(const char *name, int score)
{
    RankEntry x;
    rank_set_name(&x, name);
    x.score = score; x.diff = 1; x.my_pts = 0; x.ai_pts = 0;
    return x;
}

int main(void)
{
    const char *tmp = "ranking_test_tmp.txt";
    Ranking r, r2;
    RankEntry x;
    int i;

    /* 점수 공식 */
    CHECK(rank_calc_score(7, 42, 1, 3) == 2840);   /* (700+420+300)×2.0 */
    CHECK(rank_calc_score(3, 20, 0, 1) == 500);    /* (300+200)×1.0 */
    CHECK(rank_calc_score(7, 10, 1, 2) == 1650);   /* (700+100+300)×1.5 */

    /* 정렬: 섞인 순서로 넣어도 내림차순 */
    r.n = 0;
    { int s[] = { 300, 100, 500, 200, 400 };
      for (i = 0; i < 5; i++) { x = mk("P", s[i]); rank_insert(&r, &x); } }
    CHECK(r.n == 5);
    for (i = 1; i < r.n; i++) CHECK(r.e[i - 1].score >= r.e[i].score);

    /* 반환 인덱스 = 들어간 순위 */
    x = mk("MID", 350);  CHECK(rank_insert(&r, &x) == 2);   /* 500 400 [350] 300 ... */

    /* 동점: 먼저 세운 기록이 위 */
    x = mk("LATE", 350); CHECK(rank_insert(&r, &x) == 3);
    CHECK(strcmp(r.e[2].name, "MID") == 0 && strcmp(r.e[3].name, "LATE") == 0);

    /* 10개 초과: 꼴찌보다 낮거나 같으면 못 들고, 높으면 꼴찌를 밀어냄 */
    while (r.n < RANK_MAX) { x = mk("F", 50); rank_insert(&r, &x); }
    CHECK(!rank_qualifies(&r, 50));
    x = mk("LOW", 50);   CHECK(rank_insert(&r, &x) == -1);
    x = mk("TOP", 9999); CHECK(rank_insert(&r, &x) == 0);
    CHECK(r.n == RANK_MAX && r.e[RANK_MAX - 1].score == 50);

    /* 이름 정리 */
    rank_set_name(&x, "  홍 길동\n");          CHECK(strcmp(x.name, "홍_길동") == 0);
    rank_set_name(&x, "\n");                   CHECK(strcmp(x.name, "PLAYER") == 0);
    rank_set_name(&x, "가나다라마바사");        CHECK(strcmp(x.name, "가나다라마") == 0);  /* 3바이트×5, 글자 안 끊김 */
    rank_set_name(&x, "abcdefghijklmnopqrst"); CHECK(strlen(x.name) == RANK_NAME_LEN - 1);

    /* 저장 → 불러오기 하면 그대로 */
    CHECK(rank_save(&r, tmp));
    rank_load(&r2, tmp);
    CHECK(r2.n == r.n);
    for (i = 0; i < r.n; i++) CHECK(r2.e[i].score == r.e[i].score && strcmp(r2.e[i].name, r.e[i].name) == 0);
    remove(tmp);

    /* 파일 없으면 빈 랭킹 */
    rank_load(&r2, "no_such_file_ranking.txt");  CHECK(r2.n == 0);

    rank_print(&r, 0);
    printf(fails ? "\n%d개 실패\n" : "\n모든 테스트 통과\n", fails);
    return fails != 0;
}
