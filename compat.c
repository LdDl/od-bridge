// glibc < 2.38 compatibility shims.
// ort-sys ships a prebuilt libonnxruntime.a that references __isoc23_strto*
// symbols (C23, glibc 2.38+). On older systems these are missing, so we
// provide thin wrappers that forward to the classic POSIX versions.

#include <stdlib.h>

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}
