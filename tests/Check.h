#pragma once

// A minimal test harness, so the tests need nothing beyond the standard library.

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace check {

struct Case {
    const char* name;
    std::function<void()> body;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failures() {
    static int n = 0;
    return n;
}

struct Register {
    Register(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void fail(const char* file, int line, const std::string& what) {
    ++failures();
    std::cerr << file << ':' << line << ": " << what << '\n';
}

}  // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                             \
    static void name();                                        \
    static check::Register CHECK_CAT(reg_, name)(#name, name); \
    static void name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) check::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                 \
    do {                                                                                                      \
        const double va_ = (a), vb_ = (b);                                                                    \
        if (!(std::abs(va_ - vb_) <= (tol)))                                                                  \
            check::fail(__FILE__, __LINE__,                                                                   \
                        "CHECK_NEAR(" #a ", " #b "): " + std::to_string(va_) + " vs " + std::to_string(vb_)); \
    } while (0)
