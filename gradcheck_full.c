// Full gradient check: ALL params, many random inputs/actions, double precision.
// Reuses the real forward/backward from pong_dqn.c (included, main renamed).
#define REAL double
#define main pong_main
#include "pong_dqn.c"
#undef main
int main(void){
    rs ^= 777; int trials=40; double worst=0; long checked=0, bad=0, nz=0; double gmax=0; int worst_i=-1;
    for(int t=0;t<trials;t++){
        Net n; net_init(&n);
        REAL *p=(REAL*)&n; for(int i=0;i<NP;i++) p[i]+=0.05*(frand()*2-1);   /* nonzero biases too */
        REAL x[IN],h1[H],h2[H],q[OUT]; for(int i=0;i<IN;i++) x[i]=frange(-1,1);
        int act=rnd()%3; REAL tgt=frange(-1,1);
        forward(&n,x,h1,h2,q);
        Net g; memset(&g,0,sizeof g); REAL dq[OUT]={0,0,0}; dq[act]=q[act]-tgt;
        backward(&n,x,h1,h2,dq,&g);
        const REAL *ga=(const REAL*)&g;
        for(int i=0;i<NP;i++){
            REAL old=p[i], eps=1e-6, qa[OUT];
            p[i]=old+eps; forward(&n,x,h1,h2,qa); double lp=0.5*(qa[act]-tgt)*(qa[act]-tgt);
            p[i]=old-eps; forward(&n,x,h1,h2,qa); double lm=0.5*(qa[act]-tgt)*(qa[act]-tgt);
            p[i]=old; double num=(lp-lm)/(2*eps), an=ga[i];
            double err=fabs(num-an), den=fabs(num)+fabs(an);
            checked++; if(fabs(an)>1e-8){ nz++; if(fabs(an)>gmax) gmax=fabs(an);} if(den<1e-6){ if(err>1e-8){bad++; printf("abs mismatch idx %d num=%g an=%g\n",i,num,an);} continue; }
            double rel=err/den;
            if(rel>1e-4){ bad++; } if(rel>worst){worst=rel;worst_i=i;}
        }
        /* final grad for the unchosen actions must be exactly zero for output bias */
        for(int o=0;o<OUT;o++) if(o!=act && g.b3[o]!=0){ printf("unchosen action grad nonzero!\n"); bad++; }
    }
    printf("nonzero analytic grads=%ld/%ld  max|grad|=%.3e\n",nz,checked,gmax); printf("trials=%d params/trial=%d checked=%ld  worst rel err=%.3e (param idx %d)  >1e-4: %ld  -> %s\n",
           trials,NP,checked,worst,worst_i,bad,bad==0?"PASS":"CHECK");
    return 0;
}
