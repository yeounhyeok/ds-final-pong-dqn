// Pong + DQN, pure C (no external libs). Prototype for 자료구조 기말프로젝트.
// build: gcc -O2 -o pong_dqn pong_dqn.c -lm
// usage: ./pong_dqn gradcheck | train [steps] [seed] | eval weights.bin
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef REAL
#define REAL float   /* gradcheck is built with -DREAL=double: float + ReLU kinks give false FAIL */
#endif

/* ---------- RNG (xorshift) ---------- */
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 11); }
static REAL frand(void){ return (rnd() & 0xFFFFFF) / 16777216.0f; }          /* [0,1) */
static REAL frange(REAL a, REAL b){ return a + (b - a) * frand(); }

/* ---------- MLP: 6 -> 64 -> 64 -> 3 (ReLU), flat REAL params ---------- */
#define IN 6
#define H 64
#define OUT 3
typedef struct { REAL W1[H*IN], b1[H], W2[H*H], b2[H], W3[OUT*H], b3[OUT]; } Net;
#define NP ((int)(sizeof(Net)/sizeof(REAL)))

static void net_init(Net *n){
    REAL *p = (REAL*)n; memset(n, 0, sizeof *n);
    REAL s1 = sqrtf(2.0f/IN), s2 = sqrtf(2.0f/H), s3 = sqrtf(1.0f/H);
    for(int i=0;i<H*IN;i++) n->W1[i] = (frand()*2-1)*1.732f*s1;
    for(int i=0;i<H*H;i++)  n->W2[i] = (frand()*2-1)*1.732f*s2;
    for(int i=0;i<OUT*H;i++)n->W3[i] = (frand()*2-1)*1.732f*s3;
    (void)p;
}
static void forward(const Net *n, const REAL *x, REAL *h1, REAL *h2, REAL *q){
    for(int k=0;k<H;k++){ REAL s=n->b1[k]; for(int i=0;i<IN;i++) s+=n->W1[k*IN+i]*x[i]; h1[k]=s>0?s:0; }
    for(int j=0;j<H;j++){ REAL s=n->b2[j]; for(int k=0;k<H;k++) s+=n->W2[j*H+k]*h1[k]; h2[j]=s>0?s:0; }
    for(int o=0;o<OUT;o++){ REAL s=n->b3[o]; for(int j=0;j<H;j++) s+=n->W3[o*H+j]*h2[j]; q[o]=s; }
}
/* accumulate dLoss/dparams into g given dLoss/dq */
static void backward(const Net *n, const REAL *x, const REAL *h1, const REAL *h2, const REAL *dq, Net *g){
    REAL dz2[H], dz1[H];
    for(int o=0;o<OUT;o++){ g->b3[o]+=dq[o]; for(int j=0;j<H;j++) g->W3[o*H+j]+=dq[o]*h2[j]; }
    for(int j=0;j<H;j++){ REAL s=0; for(int o=0;o<OUT;o++) s+=dq[o]*n->W3[o*H+j]; dz2[j]=h2[j]>0?s:0; }
    for(int j=0;j<H;j++){ g->b2[j]+=dz2[j]; for(int k=0;k<H;k++) g->W2[j*H+k]+=dz2[j]*h1[k]; }
    for(int k=0;k<H;k++){ REAL s=0; for(int j=0;j<H;j++) s+=dz2[j]*n->W2[j*H+k]; dz1[k]=h1[k]>0?s:0; }
    for(int k=0;k<H;k++){ g->b1[k]+=dz1[k]; for(int i=0;i<IN;i++) g->W1[k*IN+i]+=dz1[k]*x[i]; }
}
/* Adam */
typedef struct { Net m, v; int t; } Adam;
static void adam_step(Net *n, const Net *g, Adam *a, REAL lr){
    const REAL b1=0.9f, b2=0.999f, eps=1e-8f; a->t++;
    REAL c1 = 1-powf(b1,a->t), c2 = 1-powf(b2,a->t);
    REAL *p=(REAL*)n, *m=(REAL*)&a->m, *v=(REAL*)&a->v; const REAL *gr=(const REAL*)g;
    for(int i=0;i<NP;i++){
        m[i]=b1*m[i]+(1-b1)*gr[i]; v[i]=b2*v[i]+(1-b2)*gr[i]*gr[i];
        p[i]-=lr*(m[i]/c1)/(sqrtf(v[i]/c2)+eps);
    }
}

/* ---------- Pong environment (agent = left paddle, scripted opponent = right) ---------- */
#define PH 0.10f          /* paddle half height */
#define PSPD 0.03f        /* agent paddle speed / step */
#define OSPD 0.018f       /* opponent paddle speed / step */
#define VX0 0.020f
#define VXMAX 0.050f
typedef struct { REAL bx,by,bvx,bvy,py,oy; } Env;
static void env_reset(Env *e){
    e->bx=0.5f; e->by=frange(0.2f,0.8f); e->py=0.5f; e->oy=0.5f;
    e->bvx=(rnd()&1)?VX0:-VX0; e->bvy=frange(-0.015f,0.015f);
}
static void env_obs(const Env *e, REAL *s){
    s[0]=e->bx; s[1]=e->by; s[2]=e->bvx*20; s[3]=e->bvy*20; s[4]=e->py; s[5]=e->oy;
}
static REAL clampf(REAL x,REAL a,REAL b){ return x<a?a:(x>b?b:x); }
/* returns reward; *done=1 on point end. reward: hit +0.3, opp miss +1, own miss -1 */
static REAL env_step(Env *e, int act, int *done, int *hit, int *won){
    REAL r=0; *done=0; *hit=0; *won=0;
    e->py = clampf(e->py + (act-1)*PSPD, PH, 1-PH);
    REAL d = e->by - e->oy;                     /* opponent: deadzone tracking, slower than agent */
    if(e->bvx>0 || e->bx>0.5f){ if(d>0.01f) e->oy+=OSPD; else if(d<-0.01f) e->oy-=OSPD; }
    e->oy = clampf(e->oy, PH, 1-PH);
    e->bx += e->bvx; e->by += e->bvy;
    if(e->by<0){ e->by=-e->by; e->bvy=-e->bvy; }
    if(e->by>1){ e->by=2-e->by; e->bvy=-e->bvy; }
    if(e->bx<=0){
        if(fabsf(e->by-e->py)<=PH){
            e->bx=-e->bx; e->bvx=fminf(-e->bvx*1.04f,VXMAX);
            e->bvy=clampf(e->bvy+0.02f*(e->by-e->py)/PH,-0.04f,0.04f); r=0.3f; *hit=1;
        } else { r=-1; *done=1; }
    } else if(e->bx>=1){
        if(fabsf(e->by-e->oy)<=PH){
            e->bx=2-e->bx; e->bvx=-e->bvx; e->bvy=clampf(e->bvy+0.02f*(e->by-e->oy)/PH,-0.04f,0.04f);
        } else { r=1; *done=1; *won=1; }
    }
    return r;
}

/* ---------- Replay buffer: circular queue ---------- */
typedef struct { REAL s[IN], s2[IN], r; int a, d; } Tr;
typedef struct { Tr *buf; int cap, head, size; } Replay;
static void rp_init(Replay *q,int cap){ q->buf=malloc(sizeof(Tr)*cap); q->cap=cap; q->head=0; q->size=0; }
static void rp_push(Replay *q,const Tr *t){ q->buf[q->head]=*t; q->head=(q->head+1)%q->cap; if(q->size<q->cap) q->size++; }

/* ---------- evaluation: greedy policy over N points ---------- */
static int argmax(const REAL *q){ int b=0; for(int i=1;i<OUT;i++) if(q[i]>q[b]) b=i; return b; }
typedef struct { REAL win; REAL hits; } Eval;
static Eval evaluate(const Net *n, int points, int random_policy){
    Env e; int w=0, hs=0; REAL s[IN], h1[H], h2[H], q[OUT];
    for(int p=0;p<points;p++){
        env_reset(&e); int done=0, hit, won, steps=0, ph=0;
        while(!done && steps<2000){
            env_obs(&e,s); int a;
            if(random_policy) a=rnd()%3; else { forward(n,s,h1,h2,q); a=argmax(q); }
            env_step(&e,a,&done,&hit,&won); ph+=hit; steps++;
        }
        w+=won; hs+=ph;
    }
    Eval r={(REAL)w/points,(REAL)hs/points}; return r;
}

/* ---------- gradient check (analytic vs numeric) ---------- */
static int gradcheck(void){
    Net n; net_init(&n); REAL x[IN], h1[H],h2[H],q[OUT];
    for(int i=0;i<IN;i++) x[i]=frange(-1,1);
    int act=1; REAL tgt=0.7f;
    forward(&n,x,h1,h2,q);
    Net g; memset(&g,0,sizeof g); REAL dq[OUT]={0,0,0}; dq[act]=q[act]-tgt;   /* L = 0.5 (q-t)^2 */
    backward(&n,x,h1,h2,dq,&g);
    REAL *p=(REAL*)&n; const REAL *ga=(const REAL*)&g; double maxrel=0; int checked=0;
    for(int i=0;i<NP;i+=7){
        REAL old=p[i], eps=(sizeof(REAL)==8?1e-5:1e-2), qa[OUT];
        p[i]=old+eps; forward(&n,x,h1,h2,qa); double lp=0.5*(qa[act]-tgt)*(qa[act]-tgt);
        p[i]=old-eps; forward(&n,x,h1,h2,qa); double lm=0.5*(qa[act]-tgt)*(qa[act]-tgt);
        p[i]=old; double num=(lp-lm)/(2*eps), an=ga[i];
        double den=fabs(num)+fabs(an); if(den<1e-6) continue;
        double rel=fabs(num-an)/den; if(rel>maxrel) maxrel=rel; checked++;
    }
    printf("gradcheck: %d params checked of %d, max relative error = %.2e  -> %s\n",
           checked, NP, maxrel, maxrel<1e-2?"PASS":"FAIL");
    return maxrel<1e-2?0:1;
}

/* ---------- DQN training ---------- */
static int train(long steps, unsigned long long seed, const char *out){
    rs ^= seed*0x9E3779B97F4A7C15ULL; for(int i=0;i<20;i++) rnd();
    const int B=32, WARM=2000; int TSYNC=getenv("TSYNC")?atoi(getenv("TSYNC")):500;
    const REAL GAMMA=getenv("GAMMA")?atof(getenv("GAMMA")):0.99f, LR=getenv("LR")?atof(getenv("LR")):1e-3f;
    const char *tag=getenv("TAG")?getenv("TAG"):"run"; char lgname[64]; snprintf(lgname,64,"train_log_%s.csv",tag); char wname[64]; snprintf(wname,64,"best_%s.bin",tag); out=wname;
    Net on, tg; net_init(&on); tg=on; Adam *ad=calloc(1,sizeof(Adam));
    Replay rp; rp_init(&rp,getenv("RB")?atoi(getenv("RB")):50000); int DDQN=getenv("DDQN")?atoi(getenv("DDQN")):0;
    Env e; env_reset(&e);
    Eval base=evaluate(&on,300,1); printf("baseline random policy: win=%.3f hits/point=%.2f\n",base.win,base.hits);
    FILE *lg=fopen(lgname,"w"); fprintf(lg,"step,eps,eval_win,eval_hits,sec\n");
    REAL s[IN],h1[H],h2[H],q[OUT]; REAL best=-1; clock_t t0=clock(); int ep_steps=0;
    for(long t=1;t<=steps;t++){
        REAL eps = fmaxf(0.05f, 1.0f - (REAL)t/(0.4f*steps));
        env_obs(&e,s); int a;
        if(frand()<eps) a=rnd()%3; else { forward(&on,s,h1,h2,q); a=argmax(q); }
        int done,hit,won; REAL r=env_step(&e,a,&done,&hit,&won); ep_steps++;
        Tr tr; memcpy(tr.s,s,sizeof s); env_obs(&e,tr.s2); tr.a=a; tr.r=r; tr.d=done; rp_push(&rp,&tr);
        if(done||ep_steps>=500){ env_reset(&e); ep_steps=0; }
        if(rp.size>=WARM){
            Net g; memset(&g,0,sizeof g);
            for(int b=0;b<B;b++){
                const Tr *x=&rp.buf[rnd()%rp.size]; REAL t1[H],t2[H],tq[OUT];
                forward(&on,x->s,h1,h2,q);
                REAL target=x->r;
                if(!x->d){
                    forward(&tg,x->s2,t1,t2,tq);
                    if(DDQN){ REAL oh1[H],oh2[H],oq[OUT]; forward(&on,x->s2,oh1,oh2,oq); target+=GAMMA*tq[argmax(oq)]; } /* Double DQN */
                    else target+=GAMMA*fmaxf(tq[0],fmaxf(tq[1],tq[2]));
                }
                REAL dq[OUT]={0,0,0}; REAL err=q[x->a]-target;
                dq[x->a]=clampf(err,-1,1)/B;                         /* Huber */
                backward(&on,x->s,h1,h2,dq,&g);
            }
            adam_step(&on,&g,ad,LR);
        }
        if(t%TSYNC==0) tg=on;
        if(t%20000==0){
            Eval ev=evaluate(&on,300,0); double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("step %7ld eps=%.2f  win=%.3f hits/point=%.2f  (%.0fs)\n",t,eps,ev.win,ev.hits,sec);
            fprintf(lg,"%ld,%.3f,%.3f,%.3f,%.1f\n",t,eps,ev.win,ev.hits,sec); fflush(lg);
            if(ev.win+0.001f*ev.hits>best){ best=ev.win+0.001f*ev.hits; FILE *f=fopen(out,"wb"); fwrite(&on,sizeof on,1,f); fclose(f); }
        }
    }
    fclose(lg); printf("best weights saved -> %s\n",out); return 0;
}

int main(int argc,char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s gradcheck | train [steps] [seed] | eval weights.bin\n",argv[0]); return 1; }
    if(!strcmp(argv[1],"gradcheck")){ rs^=12345; return gradcheck(); }
    if(!strcmp(argv[1],"train")) return train(argc>2?atol(argv[2]):300000, argc>3?strtoull(argv[3],0,10):1, "best.bin");
    if(!strcmp(argv[1],"eval") && argc>2){
        Net n; FILE *f=fopen(argv[2],"rb"); if(!f||fread(&n,sizeof n,1,f)!=1){ perror("weights"); return 1; } fclose(f);
        Eval ev=evaluate(&n,1000,0); Eval rn=evaluate(&n,1000,1);
        printf("trained: win=%.3f hits/point=%.2f | random: win=%.3f hits/point=%.2f\n",ev.win,ev.hits,rn.win,rn.hits); return 0;
    }
    return 1;
}
