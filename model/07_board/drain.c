// drain.c — do "con doi" cua weight_loader: bom word 0 vao window WEIGHT
// toi khi load_done len (hoac cham tran), in ra so word da bom.
//   ROM dung     : load_done som (0..vai word neu loader dang doi byte thieu)
//   ROM = 0      : load_done sau ~832,000 word (51 seg x 65536 byte)
//   khong start  : treo ngay word dau (wr_ready=0)
// Build: gcc -O2 -o drain drain.c emamba_hw.c     Chay: sudo ./drain
#include <stdio.h>
#include "emamba_hw.h"

int main(void) {
    if (emamba_open()) return 1;
    printf("STATUS truoc = 0x%08x\n", Xil_In32(R_STATUS));
    long n;
    for (n = 0; n < 1200000; n++) {
        if (Xil_In32(R_STATUS) & S_LOADDONE) break;
        Xil_Out32(WGT_OFF/4, 0);          // moi word = 4 byte vao loader
        if ((n % 100000) == 0) { printf("... %ld word\n", n); fflush(stdout); }
    }
    printf("KET QUA: %ld word bom them, STATUS = 0x%08x\n", n, Xil_In32(R_STATUS));
    return 0;
}
