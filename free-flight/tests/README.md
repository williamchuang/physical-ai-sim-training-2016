# `free-flight/tests/` — Tests of geometry, physics and gradients

```sh
cd free-flight && python -m pytest tests
```

`test_flight.py` checks that

- the strip matches its parametrisation and has one side;
- the local frame is orthonormal and the closest-point projection finds the closest point;
- the surface is solid and collected coins vanish;
- the gradients of a four-step rollout (policy → physics → loss) match finite
  differences (`torch.autograd.gradcheck`).

The transformer's tests are in [`transformer/free-flight/`](../../transformer/free-flight/).
