// emamba_hw.c — scan UIO "MY_IP" + mmap (BAN MOI: chi can reg, khong DMA/DDR).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "emamba_hw.h"

struct emamba_hw hw;

static int filter(const struct dirent *d) { return d->d_name[0] != '.'; }

static int name_is(const char *uio, const char *target) {
    char path[128], name[64];
    FILE *fp;
    snprintf(path, sizeof path, "/sys/class/uio/%s/name", uio);
    if (!(fp = fopen(path, "r"))) return 0;
    if (!fgets(name, sizeof name, fp)) { fclose(fp); return 0; }
    fclose(fp);
    name[strcspn(name, "\n")] = 0;
    return strcmp(name, target) == 0;
}

static U64 map_size(const char *uio) {
    char path[128], s[64];
    FILE *fp;
    snprintf(path, sizeof path, "/sys/class/uio/%s/maps/map0/size", uio);
    if (!(fp = fopen(path, "r"))) return 0;
    if (!fgets(s, sizeof s, fp)) { fclose(fp); return 0; }
    fclose(fp);
    return strtoull(s, NULL, 16);
}

static void *uio_mmap(const char *uio) {
    char path[64];
    U64 sz = map_size(uio);
    if (!sz) return NULL;
    snprintf(path, sizeof path, "/dev/%s", uio);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) return NULL;
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? NULL : p;
}

int emamba_open(void) {
    struct dirent **list;
    int i, n = scandir("/sys/class/uio", &list, filter, alphasort);
    if (n < 0) { perror("/sys/class/uio"); return -1; }
    memset(&hw, 0, sizeof hw);
    for (i = 0; i < n; i++) {
        if (!hw.reg && name_is(list[i]->d_name, "MY_IP"))
            hw.reg = uio_mmap(list[i]->d_name);
        free(list[i]);
    }
    free(list);
    if (!hw.reg) {
        fprintf(stderr, "Khong thay UIO \"MY_IP\" (kiem tra: cat /sys/class/uio/uio*/name)\n");
        return -1;
    }
    return 0;
}

// Barrier sau moi access -> ep thu tu (mmap Normal-NC, CPU co the reorder).
static inline void mb(void) {
    __sync_synchronize();
#ifdef __aarch64__
    __asm__ volatile("dsb sy" ::: "memory");
#endif
}
void Xil_Out32(U32 widx, U32 data) { hw.reg[widx] = data; mb(); }
U32  Xil_In32 (U32 widx)           { U32 v = hw.reg[widx]; mb(); return v; }
