# drone-fleet-ternary

**Autonomous drone fleet coordination using ternary neural networks, stigmergic pheromone fields, GPU warp-vote consensus, CRDT state synchronization, and conservation-law verification.** Each drone is modeled as a GPU thread running a 2-layer ternary network ({−1, 0, +1} weights), and fleet decisions use `__ballot_sync` semantics for O(1) consensus in constant time.

## Why It Matters

Multi-agent drone systems face three fundamental challenges: (1) individual decision-making under sensor uncertainty, (2) decentralized task allocation without a coordinator, and (3) fleet-level safety guarantees. This crate addresses all three using a unified ternary algebra:

1. **Ternary neural networks** (BitNet b1.58 architecture) replace FP32 weights with {−1, 0, +1}, enabling 16× memory density reduction and integer-only inference suitable for edge GPUs.
2. **Pheromone-based stigmergic coordination** (inspired by ant colony optimization) provides decentralized task allocation where drones follow gradient fields — no central scheduler needed.
3. **GPU warp voting** (32-thread ballot) gives O(1) constant-time consensus on fleet directives, exploiting hardware-level synchronization primitives.

The conservation verifier ensures that weight hotswaps (runtime model updates) never corrupt the task assignment invariant: Σ(coverage) + Σ(en-route) = Σ(targets).

## How It Works

### Layer 1: Ternary Neural Network (Per-Drone Inference)

Each drone runs a 2-layer ternary network:

```
Input (4 sensors) → Hidden (8 units) → Output (3 actions)
```

- **Weights**: 56 ternary values packed into 4 `u32` words (2 bits per trit, 16 trits per word)
- **Arithmetic**: Integer-only dot products using `i8` accumulation
- **Memory**: 16 bytes for all weights (vs. 224 bytes for FP32 — 14× denser)

**Quantization**: Sensor readings (continuous `f32`) are thresholded to {−1, 0, +1} via:

$$q(x) = \begin{cases} +1 & \text{if } x > 0.15 \\ -1 & \text{if } x < -0.15 \\ 0 & \text{otherwise} \end{cases}$$

**Forward pass complexity**: O(n × m) where n = layer width, m = layer depth. Here it's 4×8 + 8×3 = 56 multiply-accumulate operations — trivially fast.

### Layer 2: Pheromone Field (Stigmergic Task Allocation)

Targets emit pheromone proportional to priority. Drones sample the gradient and follow it. Pheromone evaporates at rate ρ per tick:

$$p_i(t+1) = p_i(t) \times (1 - \rho)$$

When a drone visits a target, pheromone is dampened by factor (1 − 0.3δ), deterring other drones from racing to the same location. This is the **ant colony optimization** principle applied to multi-drone task allocation.

**Big-O**: `sample_gradient` is O(T) where T = number of uncovered targets. `tick` (evaporation) is O(T).

### Layer 3: Warp-Vote Consensus (Fleet Safety)

Drones vote on directives: `ContinueMission`, `Regroup`, or `AbortReturn`. Votes are collected via GPU warp ballot (`__ballot_sync`):

- **Encoding**: Each vote maps to a Trit: `ContinueMission → +1`, `Regroup → 0`, `AbortReturn → −1`
- **Ballot**: Two `u32` bitmasks encode 32 votes in 64 bits total
- **Directive resolution**: Abort wins if ≥ ⅓ of drones vote abort (safety-first threshold). Otherwise, majority between Continue and Regroup.

**Complexity**: O(1) — the ballot and directive computation operate on two 32-bit masks regardless of fleet size (up to 32 drones per warp). Cycle cost: 16 GPU cycles (8 setup + 4 + 4 for two passes).

### Layer 4: CRDT World State

Each drone broadcasts its state (position, battery, active target, weight version) via a Last-Writer-Wins register CRDT:

$$\text{merge}(a, b) = \begin{cases} a & \text{if } ts_a \geq ts_b \\ b & \text{otherwise} \end{cases}$$

LWW registers converge because the merge is commutative, associative, and idempotent — safe to merge in any order, any number of times.

**Complexity**: `merge` is O(D) where D = number of drones. `upsert` is O(1) expected (HashMap).

### Layer 5: Conservation Verifier

Before and after a weight hotswap, the verifier checks:

$$\text{Coverage} + \text{Reachable} + \text{Assigned} \geq \lceil f_{\min} \times T \rceil$$

Where T = total targets and f_min is the minimum coverage fraction. Additionally, all drones must report the same weight version (no split-brain).

## Quick Start

```rust
use drone_fleet_ternary::*;

// Define targets
let targets: Vec<Target> = (0..5).map(|i| Target {
    id: i, x: i as f32 * 0.2, y: 0.5, priority: 1.0, covered: false,
}).collect();

// Run a mission: 8 drones, 5 targets, 30 ticks
let report = run_mission(8, targets, 30);
println!("Coverage: {}/{} in {} ticks (battery: {:.0}%)",
    report.final_covered, report.n_targets,
    report.ticks_elapsed, report.mean_battery * 100.0);
```

## API

### Ternary Types
- `Trit` — Enum: `Neg(−1)`, `Zero(0)`, `Pos(+1)`. 2-bit encoding.
- `WeightBank` — Packed ternary weights: 16 trits per `u32`. `pack`, `get`, `len`, `bytes_used`.

### Navigation
- `TernaryNavNet` — 2-layer ternary network (56 weights). `new(trits, version)`, `infer(&SensorReading) -> DroneAction`.
- `SensorReading` — Obstacle clearance, target bearing, battery SoC, speed.
- `DroneAction` — `AvoidObstacle`, `Hold`, `Advance`.

### Coordination
- `PheromoneField` — Stigmergic task market. `tick`, `sample_gradient`, `cover`, `visit`, `covered_count`.
- `DroneWarp` — 32-drone GPU warp ballot. `set_vote`, `execute_ballot -> FleetDirective`.
- `DroneWorldCRDT` — LWW-register CRDT for drone state. `upsert`, `merge`, `get`, `mean_battery`, `weight_version_consistent`.
- `ConservationVerifier` — Pre/post hotswap safety gate. `verify -> VerificationResult`.

## Architecture Notes

This crate implements the γ + η = C conservation link at the fleet level:

- **γ** (gamma) = set of targets with active pheromone + drones assigned to them
- **η** (eta) = set of covered targets (completed work)
- **C** (constant) = total target set

The invariant γ + η = C must hold before and after any weight hotswap. If it breaks, the `ConservationVerifier` returns `CoverageDeficit` or `VersionSplitBrain`, blocking the hotswap.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. Wang, H., et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." *arXiv:2310.11453.* — Ternary weight quantization for neural networks.
2. Ma, Z., et al. (2024). "Scheduling for Cellular-connected UAV Swarm: A Swarm Intelligence Approach." *IEEE Trans. Vehicular Technology.* — Pheromone-based UAV coordination.
3. NVIDIA (2024). "CUDA C++ Programming Guide: Warp Vote Functions." `__ballot_sync` documentation.
4. Shapiro, M., et al. (2011). "A Comprehensive Study of Convergent and Commutative Replicated Data Types." *INRIA RR-7506.* — LWW-register CRDT formal definition.
5. Bonabeau, E., Dorigo, M., & Theraulaz, G. (1999). *Swarm Intelligence: From Natural to Artificial Systems.* Oxford University Press.

## License

Apache-2.0
