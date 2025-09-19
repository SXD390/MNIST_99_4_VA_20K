### 🔁 Change Log (Architecture & Training Iterations)

| Iter | Params | Architecture Snapshot | Epochs | LR / Scheduler | Train Acc | Test Acc | What changed & why |
|-----:|-------:|-----------------------|-------:|----------------|-----------|----------|--------------------|
| I0   | 18,060 | Tiny CNN, 2×Conv→Pool→2×Conv→Pool, FC(1176→10) | 1 | 0.1 / StepLR | 93.0% | 98.0% | Baseline; simple & fast but underfits a bit |
| I1   | 20,978 | +BN,+Dropout; same FC head | 10 | 0.2 / OneCycle | 99.5% | 99.6% | BN stabilizes; Dropout regularizes; OneCycle speeds convergence |
| I2   | 19,494 | +1×1 transitions after pools | 12 | 0.2 / OneCycle | 99.6% | 99.6% | Channel compression improves parameter efficiency |
| I3   | 14,500 | GAP head (GAP→1×1 32→10), channels tuned | 15 | 0.2 / OneCycle | 99.7% | 99.6% | Replace heavy FC with GAP; spend params on features |
