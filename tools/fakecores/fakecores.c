/* Report a fake number of CPUs to OpenBLAS (and anything else) on macOS.
   Interposes sysctlbyname("hw.physicalcpu"/"hw.ncpu"/...) and sysconf(_SC_NPROCESSORS_*).
   Count taken from env FAKE_NCPU. Build for x86_64 (Rosetta python):
     clang -arch x86_64 -dynamiclib -o fakecores_x86_64.dylib fakecores.c
   Use: DYLD_INSERT_LIBRARIES=/abs/path/fakecores_x86_64.dylib FAKE_NCPU=10 python ... */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysctl.h>

static int fake(void) { const char *s = getenv("FAKE_NCPU"); return s ? atoi(s) : 0; }

static int my_sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen) {
    int n = fake();
    if (n > 0 && oldp && oldlenp && *oldlenp >= sizeof(int) &&
        (!strcmp(name, "hw.physicalcpu") || !strcmp(name, "hw.logicalcpu") || !strcmp(name, "hw.ncpu") ||
         !strcmp(name, "hw.physicalcpu_max") || !strcmp(name, "hw.logicalcpu_max") || !strcmp(name, "hw.activecpu"))) {
        *(int *)oldp = n; *oldlenp = sizeof(int); return 0;
    }
    return sysctlbyname(name, oldp, oldlenp, newp, newlen);
}
static long my_sysconf(int name) {
    int n = fake();
    if (n > 0 && (name == _SC_NPROCESSORS_CONF || name == _SC_NPROCESSORS_ONLN)) return n;
    return sysconf(name);
}
__attribute__((used)) static struct { const void *rep; const void *orig; } interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
    { (const void *)my_sysctlbyname, (const void *)sysctlbyname },
    { (const void *)my_sysconf, (const void *)sysconf },
};
