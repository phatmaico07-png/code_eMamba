// memtest.c — ghi/đọc thanh ghi qua /dev/mem (O_SYNC = Device memory, KHÔNG qua UIO).
//   Phân định: nếu ghi 0x44/0x48 (không 16-aligned) ĐỌC LẠI ĐÚNG -> lỗi là mmap UIO.
// Build: gcc -O2 -o memtest memtest.c     Chay: sudo ./memtest
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define BASE 0xA0000000UL
#define SIZE 0x10000UL

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 1; }
    volatile uint32_t *r = mmap(NULL, SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BASE);
    if (r == MAP_FAILED) { perror("mmap"); return 1; }

    struct { unsigned off; uint32_t val; } t[] = {
        {0x40, 0x11111111}, {0x44, 0x22222222}, {0x48, 0x33333333},
        {0x4c, 0x44444444}, {0x50, 0x55555555},
    };
    printf("== ghi/doc qua /dev/mem (Device memory) ==\n");
    for (int i = 0; i < 5; i++) {
        r[t[i].off/4] = t[i].val;
        __sync_synchronize();
#ifdef __aarch64__
        __asm__ volatile("dsb sy" ::: "memory");
#endif
    }
    int ok = 1;
    for (int i = 0; i < 5; i++) {
        uint32_t got = r[t[i].off/4];
        printf("  0x%03x: ghi %08x doc %08x  %s\n",
               t[i].off, t[i].val, got, got==t[i].val ? "OK" : "SAI");
        if (got != t[i].val) ok = 0;
    }
    printf(ok ? ">> TAT CA OK -> loi nam o mmap UIO\n"
              : ">> Co SAI -> loi sau hon (FPGA/interconnect)\n");
    munmap((void*)r, SIZE); close(fd);
    return 0;
}
