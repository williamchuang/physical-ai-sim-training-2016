#include "Check.h"

int main() {
    for (const check::Case& c : check::registry()) {
        const int before = check::failures();
        c.body();
        std::cout << (check::failures() == before ? "  ok    " : "  FAIL  ") << c.name << '\n';
    }
    std::cout << check::registry().size() << " tests, " << check::failures() << " failed checks\n";
    return check::failures() == 0 ? 0 : 1;
}
