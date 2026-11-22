# `transformer/config/` — Settings to run, record and evaluate the transformer

Settings files for the restored simulation, run from the repository root.

| File | |
|---|---|
| `demo.cfg` | the transformer bot collects coins on its own |
| `compare.cfg` | you on the keys beside the transformer bot |
| `evaluate.cfg` | one unattended 90 s round; prints the score and quits (add `seed = N`) |
| `record.cfg` | records camera frames and the autopilot's keys to `transformer/recordings/` for training; add `policy_kind = transformer` for a DAgger round |

```sh
./build/pai_sim transformer/config/demo.cfg
```

The keys these files use: `policy_kind` (`mlp` or `transformer`), `transformer_file`
(default `transformer/models/transformer.txt`), `history_frames` (4),
`decision_interval` (0.05, the rate the transformer was trained at), `record_file`,
and `fixed_frame_seconds` (simulated seconds per frame; 0 = real time).
