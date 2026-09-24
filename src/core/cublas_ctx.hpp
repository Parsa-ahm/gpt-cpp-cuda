#pragma once
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

#define CUBLAS_CHECK(expr)                                                                        \
    do {                                                                                          \
        cublasStatus_t st_ = (expr);                                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n  in: %s\n", __FILE__, __LINE__, \
                         (int)st_, #expr);                                                        \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

inline cublasHandle_t cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr)
        CUBLAS_CHECK(cublasCreate(&h));
    return h;
}
