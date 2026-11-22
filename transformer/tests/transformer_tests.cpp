// The transformer's C++ forward pass against PyTorch.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Check.h"
#include "Transformer.h"

using namespace pai;

#ifndef PAI_TRANSFORMER_TEST_DATA
#define PAI_TRANSFORMER_TEST_DATA "transformer/tests"
#endif

// The C++ forward pass must give PyTorch's outputs for PyTorch's weights
// (fixture written by transformer/training/make_fixture.py).
TEST(transformer_matches_pytorch) {
    std::ifstream in(std::string(PAI_TRANSFORMER_TEST_DATA) + "/transformer_fixture.txt");
    CHECK(static_cast<bool>(in));
    const Transformer net(in);
    std::string tag;
    std::size_t cases = 0;
    in >> tag >> cases;
    CHECK(tag == "cases" && cases > 0);
    double worst = 0.0;
    for (std::size_t c = 0; c < cases; ++c) {
        std::vector<float> x(net.shape().inputs());
        for (float& v : x) in >> v;
        std::vector<double> expected(net.shape().outputs);
        for (double& v : expected) in >> v;
        const std::vector<float> got = net.logits(x);
        for (std::size_t i = 0; i < expected.size(); ++i)
            worst = std::max(worst, std::abs(got[i] - expected[i]) / (1.0 + std::abs(expected[i])));
    }
    CHECK(static_cast<bool>(in));
    std::cout << "    largest difference from PyTorch: " << worst << '\n';
    CHECK(worst < 1e-4);
}

TEST(transformer_rejects_wrong_shapes) {
    std::istringstream bad("transformer 1 4 20 20 6 32 2 2 64 4\n");  // 6 does not divide 20
    bool threw = false;
    try {
        Transformer t(bad);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}
