#define main orig_main
#include "../pong_dqn_v2.c"
#undef main
static Net load(const char*p){ Net n; FILE*f=fopen(p,"rb"); if(!f||fread(&n,sizeof n,1,f)!=1){exit(1);} fclose(f); return n; }
static void seedit(unsigned long long x){ rs = x*2654435761ULL+12345; for(int i=0;i<30;i++)rnd(); }
static REAL gauss(void){ REAL u=(frand()+1e-6f),v=frand(); return sqrtf(-2*logf(u))*cosf(6.2831853f*v); }
/* knob: eps=무작위 행동 확률, sig=착지 예측 입력(s[6])에 더하는 가우시안 오차, hold=행동을 hold 스텝마다만 갱신(반응 지연) */
static REAL run(const Net*n,int pts,REAL eps,REAL sig,int hold){
  Env e; int wins=0; REAL s[IN],h1[H],h2[H],q[OUT];
  for(int p=0;p<pts;p++){ env_reset(&e); int done=0,hit,won=0,st=0,a=1;
    while(!done&&st<2000){ if(st%hold==0){ env_obs(&e,s); s[6]=clampf(s[6]+sig*gauss(),0,1);
        if(frand()<eps) a=rnd()%3; else { forward(n,s,h1,h2,q); a=argmax(q);} }
      env_step(&e,a,&done,&hit,&won); st++; }
    wins+=won; }
  return (REAL)wins/pts; }
int main(void){ Net n=load("models/final_model.bin"); int P=5000;
  printf("== eps(무작위 행동) ==\n"); REAL E[]={0,0.05,0.1,0.2,0.3,0.4,0.5,0.7};
  for(int i=0;i<8;i++){ seedit(31); printf("eps=%.2f win=%.4f\n",E[i],run(&n,P,E[i],0,1)); }
  printf("== sig(착지 예측 오차) ==\n"); REAL S[]={0.02,0.05,0.1,0.15,0.2,0.3};
  for(int i=0;i<6;i++){ seedit(31); printf("sig=%.2f win=%.4f\n",S[i],run(&n,P,0,S[i],1)); }
  printf("== hold(행동 갱신 주기) ==\n"); int H2[]={2,4,8,12,16,24};
  for(int i=0;i<6;i++){ seedit(31); printf("hold=%d win=%.4f\n",H2[i],run(&n,P,0,0,H2[i])); }
  return 0; }
