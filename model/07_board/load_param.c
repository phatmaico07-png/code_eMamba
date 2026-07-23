// load_param.c — nạp tham số (shift + weight) qua SINGLE-PORT STREAMING.
//   Mọi ghi vào DUY NHẤT port 0x100: lệnh rồi data. Chạy 1 lần sau mỗi nạp bitstream.
// Build: make    Chạy: sudo ./load_param weights.bin
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "emamba_hw.h"

static const U8 SH_B0[18] = {5,7,7,6,8,15,4,5,5,11,0,0,0,0,6,7,8,8};
static const U8 SH_B1[18] = {4,6,7,7,6,15,4,5,5,12,1,0,0,0,6,7,8,8};

static void pack_shift(const U8 *sh, U32 *w) {
    __uint128_t s = 0; int i;
    for (i = 0; i < 18; i++) s = (s << 5) | (sh[i] & 0x1f);
    w[0] = (U32)s;  w[1] = (U32)(s >> 32);  w[2] = (U32)(s >> 64) & 0x03ffffff;
}

int main(int argc, char **argv) {
    int i;
    if (argc < 2) { fprintf(stderr, "Dung: %s weights.bin\n", argv[0]); return 1; }
    if (emamba_open()) return 1;

    // ── 1) shift: CMD_SHIFT + 6 từ (b0[0..2], b1[0..2]) + CMD_SHCTRL ──
    U32 w0[3], w1[3];
    pack_shift(SH_B0, w0);  pack_shift(SH_B1, w1);
    CMD(CMD_SHIFT, 6);
    PUT(w0[0]); PUT(w0[1]); PUT(w0[2]);
    PUT(w1[0]); PUT(w1[1]); PUT(w1[2]);
    CMD(CMD_SHCTRL, 0);
    printf("Shift calib: OK.\n");

    // ── 2) weight: CMD_WEIGHT(3693) + 3693 từ ──
    static U8 buf[BLOB_PAD];
    memset(buf, 0, sizeof buf);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    size_t nb = fread(buf, 1, BLOB_BYTES, f);
    fclose(f);
    if (nb != BLOB_BYTES) { fprintf(stderr, "Loi: doc %zu/%d byte\n", nb, BLOB_BYTES); return 1; }

    CMD(CMD_WEIGHT, BLOB_PAD/4);                  // 3693 từ
    for (i = 0; i < BLOB_PAD/4; i++)
        PUT(buf[4*i] | (buf[4*i+1]<<8) | (buf[4*i+2]<<16) | ((U32)buf[4*i+3]<<24));
    printf("Nap %d byte weight (stream qua port).\n", BLOB_PAD);

    // ── 3) chờ load_done (đọc status) ──
    CMD(CMD_RDSTAT, 0);                            // ép read_sel=STATUS (lần chạy trước có thể để OUTPUT)
    long spins = 100000000L;
    U32 st = 0;
    while (--spins) { st = GET(); if (st & S_LOADDONE) break; }
    if (spins <= 0) {
        fprintf(stderr, "TIMEOUT load_done (status=0x%08x: wmode=%u wptr=%u wrem=%u)\n",
                st, (st>>12)&7, (st>>4)&0x7f, (st>>16)&0xffff);
        return 1;
    }
    printf("load_done = 1 -> tham so OK (status=0x%08x)\n", st);

    // ── verify dữ liệu đã nạp (đọc lại từ phần cứng) ──
    CMD(CMD_RDSUM, 0);
    U32 ws = GET();
    printf("WSUM doc lai = 0x%08X  (ky vong 0xBE8FCF11) -> %s\n",
           ws, ws == 0xBE8FCF11u ? "KHOP" : "*** SAI ***");
    CMD(CMD_RDSHIFT, 0);
    U32 s0 = GET(), s1 = GET(), s2 = GET(), s3 = GET(), s4 = GET(), s5 = GET();
    printf("SHIFT doc lai b0= %08X %08X %08X\n", s0, s1, s2);
    printf("            b1= %08X %08X %08X\n", s3, s4, s5);
    printf("  ky vong   b0= 00031D08 F214AB00 00A73990\n");
    printf("            b1= 00031D08 F214AC08 008639CC\n");
    {
        int ok = (s0==0x00031D08u && s1==0xF214AB00u && s2==0x00A73990u &&
                  s3==0x00031D08u && s4==0xF214AC08u && s5==0x008639CCu);
        printf("SHIFT -> %s\n", ok ? "KHOP" : "*** SAI ***");
    }

    printf("Chay: sudo ./emamba input.bin output.bin [golden.bin]\n");
    return 0;
}
