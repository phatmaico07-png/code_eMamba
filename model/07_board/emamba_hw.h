// emamba_hw.h — driver UIO cho eMamba (KV260).
//   *** SINGLE-PORT streaming (KV26 chi 1 o dia chi tin cay), weight NUNG SAN ***
//   -> KHONG nap weight/bias/shift (da bake LOAD_MODE=0). Moi giao tiep qua PORT 0x100.
#ifndef EMAMBA_HW_H
#define EMAMBA_HW_H
#include <stdint.h>
#include <stdio.h>

typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;

#define REG_BASE_PHYS  0x00000000A0000000ULL   // emamba_axi (UIO "MY_IP")

// ── SINGLE-PORT: moi access qua 1 o (byte 0x100 = word index 0x40) ──
#define PORT       (0x100/4)
#define PUT(v)     Xil_Out32(PORT, (U32)(v))
#define CMD(c,n)   Xil_Out32(PORT, ((U32)(c)<<24) | ((U32)(n) & 0x00FFFFFF))
#define GET()      Xil_In32(PORT)

// command codes (khop emamba_axi.v) — KHONG con WEIGHT/SHIFT (nung san)
#define CMD_INPUT   0x02
#define CMD_START   0x05
#define CMD_READOUT 0x06
#define CMD_RDSTAT  0x07

#define S_DONE     0x1
#define S_BUSY     0x2

#define IN_WORDS   80
#define OUT_WORDS  15
#define OUT_BYTES  57

struct emamba_hw { volatile U32 *reg; };
extern struct emamba_hw hw;

int  emamba_open(void);
void Xil_Out32(U32 widx, U32 data);
U32  Xil_In32 (U32 widx);

// ── 1 lan thu: nap input + START(re-arm) + cho done + doc output. 0=OK, 2=stall ──
static inline int emamba_run_once(const U32 *in80, U32 *out15) {
    int i, t; long sp;
    CMD(CMD_INPUT, IN_WORDS);
    for (i = 0; i < IN_WORDS; i++) PUT(in80[i]);
    for (t = 0; t < 16; t++) {                              // START + re-arm neu done khong clear
        CMD(CMD_START, 0);
        sp = 2000000L; while ( (GET() & S_DONE) && --sp) ;  // cho done 1 -> 0
        if (sp > 0) break;                                  // da clear -> START nhan
    }
    sp = 5000000L; while (!(GET() & S_DONE) && --sp) ; if (sp <= 0) return 2;  // done 0 -> 1
    CMD(CMD_READOUT, 0);
    for (i = 0; i < OUT_WORDS; i++) out15[i] = GET();
    return 0;
}

// ── drain: bom IN_WORDS word 0 -> ep wmode ve IDLE (xoa desync do rot write AXI) ──
static inline void emamba_drain(void) {
    int i; for (i = 0; i < IN_WORDS; i++) PUT(0); CMD(CMD_RDSTAT, 0);
}

// ── chay 1 frame ROBUST: duong nhanh; stall hiem -> drain don desync roi thu lai ──
//   Tra ve: 0=OK, 2=stall that su (sau 8 lan thu) ──
static inline int emamba_run_frame(const U32 *in80, U32 *out15) {
    int t;
    for (t = 0; t < 8; t++) {
        if (emamba_run_once(in80, out15) == 0) return 0;
        emamba_drain();
    }
    return 2;
}

#endif
