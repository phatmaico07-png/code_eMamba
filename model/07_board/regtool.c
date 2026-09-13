// regtool.c — doc/ghi 1 thanh ghi emamba qua UIO (debug).
// Build:  gcc -O2 -o regtool regtool.c emamba_hw.c
// Doc :   sudo ./regtool 0x4
// Ghi :   sudo ./regtool 0x40 0x12345678
#include <stdio.h>
#include <stdlib.h>
#include "emamba_hw.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Dung: %s <byte_off_hex> [data_hex]\n", argv[0]); return 1; }
    if (emamba_open()) return 1;
    U32 boff = (U32)strtoul(argv[1], 0, 16);
    if (argc > 2) {
        U32 d = (U32)strtoul(argv[2], 0, 16);
        Xil_Out32(boff/4, d);
        printf("[0x%04x] <= 0x%08x\n", boff, d);
    } else {
        printf("[0x%04x] = 0x%08x\n", boff, Xil_In32(boff/4));
    }
    return 0;
}
