// main.c — chay eMamba theo frame qua AXI single-port (weight NUNG SAN).
//   *** DINH DANG TXT (hex byte), PARSE NHANH (doc ca file vao RAM) ***
//   input.txt : 320 hex byte/dong (1 frame = 80 word)
//   output.txt: 57 hex byte/dong (ghi ra)
//   golden.txt: (tuy chon) 57 hex byte/dong -> so tung byte -> TONG lech X/Y PASS/FAIL
// Build: make    Chay: sudo ./emamba input.txt output.txt [golden.txt]
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "emamba_hw.h"

#define IN_BYTES  (IN_WORDS*4)   // 320
static const char HEXD[] = "0123456789abcdef";

// doc ca file vao RAM (buf cap-phat, them '\0')
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char*)malloc(sz + 1); if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, sz, f) != (size_t)sz) { /* van xu ly phan doc duoc */ }
    b[sz] = '\0'; fclose(f); return b;
}
static int hx(int c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10;
                      if(c>='A'&&c<='F')return c-'A'+10; return -1; }
// parse 1 byte hex tu **p (bo qua space/newline); -1 neu het
static int next_byte(char **p){
    char *s=*p; while(*s==' '||*s=='\n'||*s=='\r'||*s=='\t') s++;
    int h=hx((unsigned char)*s); if(h<0){ *p=s; return -1; }
    int v=h; s++; int l=hx((unsigned char)*s); if(l>=0){ v=v*16+l; s++; }
    *p=s; return v;
}
// doc n byte hex; tra ve so byte parse duoc
static int read_n(char **p, U8 *buf, int n){
    int i,v; for(i=0;i<n;i++){ v=next_byte(p); if(v<0) return i; buf[i]=(U8)v; } return n;
}

int main(int argc, char **argv) {
    int i;
    if (argc < 3) { fprintf(stderr, "Dung: %s input.txt output.txt [golden.txt]\n", argv[0]); return 1; }
    if (emamba_open()) return 1;

    char *inbuf = slurp(argv[1]); if (!inbuf) return 1;
    char *gbuf  = (argc > 3) ? slurp(argv[3]) : NULL;
    if (argc > 3 && !gbuf) return 1;
    FILE *outf = fopen(argv[2], "w"); if (!outf) { perror(argv[2]); return 1; }
    char *pin = inbuf, *pg = gbuf;

    U8  inb[IN_BYTES], outb[OUT_WORDS*4], gb[OUT_BYTES];
    U32 in32[IN_WORDS], out32[OUT_WORDS];
    long f = 0, mism = 0, total = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (read_n(&pin, inb, IN_BYTES) == IN_BYTES) {
        for (i = 0; i < IN_WORDS; i++)
            in32[i] = inb[4*i] | (inb[4*i+1]<<8) | (inb[4*i+2]<<16) | ((U32)inb[4*i+3]<<24);

        int st = emamba_run_frame(in32, out32); // nap input -> START -> poll done -> doc output
        if (st) { fprintf(stderr, "\nSTALL frame %ld: done %s -> kiem tra bitstream/driver single-port\n",
                          f, st==1?"khong clear":"khong set"); break; }

        for (i = 0; i < OUT_WORDS; i++) {
            U32 v = out32[i];
            outb[4*i] = v;  outb[4*i+1] = v>>8;  outb[4*i+2] = v>>16;  outb[4*i+3] = v>>24;
        }
        { char ln[OUT_BYTES*3]; for (i=0;i<OUT_BYTES;i++){ ln[3*i]=HEXD[outb[i]>>4]; ln[3*i+1]=HEXD[outb[i]&15]; ln[3*i+2]=(i==OUT_BYTES-1)?'\n':' '; } fwrite(ln,1,OUT_BYTES*3,outf); }

        if (gbuf && read_n(&pg, gb, OUT_BYTES) == OUT_BYTES) {
            for (i = 0; i < OUT_BYTES; i++) if (outb[i] != gb[i]) mism++;
            total += OUT_BYTES;
        }
        f++;
        if ((f % 1000) == 0) { printf("\r  ...%ld frame", f); fflush(stdout); }
    }
    printf("\r                         \r");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    printf("Xong %ld frame trong %.3f s  (%.1f us/frame, %.0f fps)\n",
           f, sec, f ? sec*1e6/f : 0.0, f ? f/sec : 0.0);
    if (gbuf) printf("TONG lech %ld/%ld  -> %s\n", mism, total, mism ? "FAIL" : "PASS");

    fclose(outf); free(inbuf); if (gbuf) free(gbuf);
    return (gbuf && mism) ? 1 : 0;
}
