# Dummy fixtures

These fixtures provide a tiny, deterministic intermediate representation for tests.

## Layout

- `ir_clean/`: Raw IR inputs with a single missing value in `speed` for node `B` at `2024-01-01 02:00:00`.
- `ir_expected/fill_zeros/`: Expected output after applying `FillZeros` (missing value becomes `0.0` plus a mask).
- `ir_expected/forward_fill/`: Expected output after applying `ForwardFill` (missing value becomes previous value `20` plus a mask).

The data uses two nodes, five hourly timestamps, and two feature columns (`speed`, `volume`) to keep hand-verified expectations simple.
