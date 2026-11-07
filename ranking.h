/* ranking.h — 랭킹 TOP 10 · 점수 기록 모듈 (담당: 서태호)
 *
 * 점수 = (내 득점 × 100 + 받아친 횟수 × 10 + 승리 보너스 300) × 난이도 배율(1.0 / 1.5 / 2.0)
 *
 * 자료구조: 크기 RANK_MAX 의 배열을 점수 내림차순으로 유지한다.
 *   - 새 기록은 맨 뒤에 넣고 삽입 정렬의 한 단계(앞으로 밀어 올리기)로 제자리를 찾는다.
 *   - 비교를 '<' 로만 하므로 안정 정렬 → 동점이면 먼저 세운 기록이 위에 남는다.
 * 저장: 텍스트 파일 한 줄에 한 기록 "이름 점수 난이도 내득점 AI득점" (프로그램을 꺼도 유지)
 * 외부 라이브러리 없이 표준 C 만 사용 → Linux(gcc) / Windows(Visual Studio) 모두 빌드 가능. */
#ifndef RANKING_H
#define RANKING_H

#define RANK_MAX       10            /* TOP 10 */
#define RANK_NAME_LEN  16            /* 이름 최대 15바이트 + '\0' (한글은 3바이트/글자 → 5글자) */
#define RANK_FILE      "ranking.txt" /* 기본 저장 파일 */

/* 점수 계산 상수 — 밸런스를 바꾸려면 여기만 고치면 된다 */
#define SCORE_PER_POINT  100         /* 내 득점 1점당 */
#define SCORE_PER_HIT     10         /* 받아친 횟수 1회당 */
#define SCORE_WIN_BONUS  300         /* 이겼을 때 */
/* 난이도 배율은 ranking.c 의 DIFF_MULT_X10 (×10 정수: 10/15/20) */

typedef struct {
    char name[RANK_NAME_LEN];
    int  score;
    int  diff;     /* 난이도 1~3 */
    int  my_pts;   /* 내 득점 */
    int  ai_pts;   /* AI 득점 */
} RankEntry;

typedef struct {
    RankEntry e[RANK_MAX];   /* e[0] 이 1등 */
    int n;                   /* 현재 기록 수 (0 ~ RANK_MAX) */
} Ranking;

/* 한 판의 점수 계산 */
int  rank_calc_score(int my_pts, int hits, int won, int diff);

/* 파일에서 불러오기 (파일이 없으면 빈 랭킹). 순서가 섞여 있어도 정렬해서 담는다. */
void rank_load(Ranking *r, const char *path);
/* 파일로 저장. 성공 1, 실패 0 */
int  rank_save(const Ranking *r, const char *path);

/* 이 점수가 TOP 10 에 들 수 있는지 (동점이면 기존 기록 우선이라 못 듦) */
int  rank_qualifies(const Ranking *r, int score);
/* 기록 삽입. 들어간 순위의 인덱스(0 = 1등)를 반환, 순위 밖이면 -1 */
int  rank_insert(Ranking *r, const RankEntry *x);

/* 입력받은 이름을 저장 가능한 형태로 정리 (공백→'_', 줄바꿈 제거, 길이 제한, 빈 이름→PLAYER) */
void rank_set_name(RankEntry *x, const char *raw);

/* 랭킹 표 출력. highlight = 강조할 인덱스 (-1 이면 강조 없음) */
void rank_print(const Ranking *r, int highlight);

/* 게임 종료 후 한 번에 처리: 점수 계산·출력 → TOP10 이면 이름 입력 → 저장 → 랭킹 표(방금 기록 강조) */
void rank_after_game(const char *path, int my_pts, int ai_pts, int hits, int diff);

#endif
