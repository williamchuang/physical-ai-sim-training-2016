# `transformer/tests/` — C++ check against PyTorch

| File | |
|---|---|
| `transformer_tests.cpp` | loads PyTorch's weights into the C++ transformer and compares the outputs for the same inputs; also rejects inconsistent shapes. Part of `pai_tests` |
| `transformer_fixture.txt` | a random transformer (3 frames, 10 x 10, 2 layers, 2 heads), 5 inputs and PyTorch's outputs, written by `../training/make_fixture.py` |

Largest relative difference between C++ and PyTorch: 2.2 × 10⁻⁷.
