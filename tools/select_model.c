#define main orig_main
#include "../pong_dqn_v2.c"
#undef main
static Net load(const char*p){ Net n; FILE*f=fopen(p,"rb"); if(!f||fread(&n,sizeof n,1,f)!=1){fprintf(stderr,"load fail %s\n",p);exit(1);} fclose(f); return n; }
static void seedit(unsigned long long x){ rs = x*2654435761ULL+12345; for(int i=0;i<30;i++)rnd(); }
int main(int argc,char**argv){
  /* argv: 후보 체크포인트들. 검증 시드 777 에서 고르고, 시험 시드 999 에서 한 번만 보고 */
  int bi=-1; double bv=-1; double v[16];
  for(int a=1;a<argc;a++){ Net n=load(argv[a]); seedit(777); Eval e=evaluate(&n,10000,0); v[a]=e.win; printf("VAL  %-22s %.4f\n",argv[a],e.win); if(e.win>bv){bv=e.win;bi=a;} }
  printf("선택: %s (검증 %.4f)\n",argv[bi],bv);
  Net n=load(argv[bi]); seedit(999); Eval t=evaluate(&n,10000,0); printf("TEST %s win=%.4f hits=%.2f (10000점, 시드 999)\n",argv[bi],t.win,t.hits);
  seedit(999); Eval r=evaluate(&n,10000,1); printf("random 정책: %.4f\n",r.win);
  return 0; }
