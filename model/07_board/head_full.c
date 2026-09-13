#include <stdio.h>
#include "head.c"          /* head57() + head_const.h (khong dinh HEAD_TEST) */
int main(int argc, char**argv){
    const char*inf  = argc>1 ? argv[1] : "feat_pwl.txt";
    const char*outf = argc>2 ? argv[2] : "poses_pwl.txt";
    FILE*f=fopen(inf,"r"); if(!f){printf("no %s\n",inf);return 1;}
    int NF; if(fscanf(f,"%d",&NF)!=1) return 1;
    FILE*o=fopen(outf,"w");
    static signed char feat[HD_L][HD_D]; signed char out[HD_OUT];
    int fr,t,d,j,v;
    for(fr=0;fr<NF;fr++){
        for(t=0;t<HD_L;t++) for(d=0;d<HD_D;d++){ if(fscanf(f,"%d",&v)!=1) return 1; feat[t][d]=(signed char)v; }
        head57(feat,out);
        for(j=0;j<HD_OUT;j++) fprintf(o,"%d ",out[j]);
        fprintf(o,"\n");
    }
    fclose(f); fclose(o);
    printf("head57 (C) chay %d frame: %s -> %s\n",NF,inf,outf);
    return 0;
}
