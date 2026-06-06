# drone-fleet-ternary

Autonomous drone fleet coordination using ternary decision-making, warp-vote consensus, CRDT state sync, and conservation verification. Each drone is a GPU node running ternary neural networks.

## Why This Matters

# drone-fleet-ternary
Autonomous drone fleet coordination using the full oxide stack:
```text
Sensor data (obstacle dist, target bearing, battery)
│

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Trit
pub fn to_i8
pub fn to_bits
pub fn from_bits
pub struct WeightBank
pub fn pack
pub fn get
pub fn len
pub fn bytes_used
pub fn bytes_fp32_equivalent
pub struct SensorReading
pub enum DroneAction
```

## Usage

```toml
[dependencies]
drone-fleet-ternary = "0.1.0"
```

```rust
use drone_fleet_ternary::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/drone-fleet-ternary.git
cd drone-fleet-ternary
cargo test    # 23 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 23 |
| Lines of Rust | 1042 |
| Public API | 48 items |

## License

Apache-2.0
