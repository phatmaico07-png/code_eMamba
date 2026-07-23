// head.c — HEAD eMamba (HƯỚNG A) chạy trên PS/CPU.
//   Nhận FEATURE backbone 16×20 INT8 (accelerator xuất) -> mean-pool -> FFN(in->ReLU->out)
//   + residual -> proj -> 57 INT8 (qo). Host nhân s_o để ra toạ độ. Toán nguyên, shift pow2,
//   KHỚP int_reference.head (golden). Model/dataset khác chỉ đổi head_const.h + hàm này.
#include "head_const.h"

static long rshr(long x, int k) {              // round-half-up arithmetic shift (= int_reference.rsh)
    if (k > 0) return (x + (1L << (k - 1))) >> k;
    if (k < 0) return x << (-k);
    return x;
}
static signed char c8(long v) { return v > 127 ? 127 : (v < -128 ? -128 : (signed char)v); }

void head57(const signed char feat[HD_L][HD_D], signed char out[HD_OUT]) {
    signed char qp[HD_D], qh[HD_H], qt[HD_D];
    int d, o, i, t;
    // mean-pool 16 token
    for (d = 0; d < HD_D; d++) {
        long s = 0; for (t = 0; t < HD_L; t++) s += feat[t][d];
        qp[d] = c8(rshr(s, SH_POOL));
    }
    // ffn_in 20->80 + bias + requant + ReLU
    for (o = 0; o < HD_H; o++) {
        long a = FFI_B[o]; for (i = 0; i < HD_D; i++) a += (long)qp[i] * FFI_W[o][i];
        long r = c8(rshr(a, SH_FFI)); qh[o] = r < 0 ? 0 : (signed char)r;
    }
    // ffn_out 80->20 + bias + requant, rồi + residual qp
    for (o = 0; o < HD_D; o++) {
        long a = FFO_B[o]; for (i = 0; i < HD_H; i++) a += (long)qh[i] * FFO_W[o][i];
        long fo = c8(rshr(a, SH_FFO)); qt[o] = c8(fo + rshr(qp[o], SH_RES));
    }
    // proj 20->57
    for (o = 0; o < HD_OUT; o++) {
        long a = PROJ_B[o]; for (i = 0; i < HD_D; i++) a += (long)qt[i] * PROJ_W[o][i];
        out[o] = c8(rshr(a, SH_PROJ));
    }
}

#ifdef HEAD_TEST
#include <stdio.h>
#include "head_test.h"
int main(void) {
    int bad = 0, k, o;
    for (k = 0; k < NT; k++) {
        signed char out[HD_OUT]; head57(FEAT[k], out);
        for (o = 0; o < HD_OUT; o++) if (out[o] != GQO[k][o]) bad++;
    }
    printf("HEAD C vs golden int_reference.head : %d/%d sai\n", bad, NT * HD_OUT);
    printf(bad ? "  FAIL\n" : "  PASS — head C BIT-EXACT.\n");
    return bad ? 1 : 0;
}
#endif
