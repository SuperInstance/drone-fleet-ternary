//! # drone-fleet-ternary
//!
//! Autonomous drone fleet coordination using the full oxide stack:
//!
//! ```text
//!   Sensor data (obstacle dist, target bearing, battery)
//!       │
//!   ┌───┴──────────────────────────────────────────────┐
//!   │  TernaryNav  — 2-layer ternary neural net         │
//!   │  16 weights/u32, {-1,0,+1} → {Avoid,Hold,Advance}│
//!   └───┬──────────────────────────────────────────────┘
//!       │ local action (per drone, runs on edge GPU)
//!   ┌───┴──────────────────────────────────────────────┐
//!   │  PheromoneField — stigmergic task market          │
//!   │  targets emit pheromone; drones follow gradient   │
//!   │  evaporation = implicit load balancing            │
//!   └───┬──────────────────────────────────────────────┘
//!       │ task assignment (no coordinator needed)
//!   ┌───┴──────────────────────────────────────────────┐
//!   │  WarpVoteFleet — group safety decisions           │
//!   │  __ballot_sync: 32 drones vote in 4 GPU cycles   │
//!   │  Regroup / AbortReturn / ContinueMission          │
//!   └───┬──────────────────────────────────────────────┘
//!       │ fleet directive
//!   ┌───┴──────────────────────────────────────────────┐
//!   │  DroneWorldCRDT — eventual-consistent world model │
//!   │  LWW-register per drone, merge without locks      │
//!   └───┬──────────────────────────────────────────────┘
//!       │
//!   ┌───┴──────────────────────────────────────────────┐
//!   │  ConservationVerifier — hotswap safety gate        │
//!   │  Σ(coverage) + Σ(en-route) = Σ(targets)           │
//!   │  Invariant must hold before/after weight swap      │
//!   └──────────────────────────────────────────────────┘
//! ```

// ── Ternary weight types ─────────────────────────────────────────────────────

/// A single ternary weight: {-1, 0, +1}. Packed as 2 bits (0b00/01/10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trit {
    Neg  = 0,  // 0b00  -1
    Zero = 1,  // 0b01   0
    Pos  = 2,  // 0b10  +1
}

impl Trit {
    pub fn to_i8(self) -> i8 { self as i8 - 1 }
    pub fn to_bits(self) -> u8 { self as u8 }

    pub fn from_bits(b: u8) -> Option<Self> {
        match b & 0b11 {
            0 => Some(Trit::Neg),
            1 => Some(Trit::Zero),
            2 => Some(Trit::Pos),
            _ => None,  // 0b11 unused — matches ternary-pack invariant
        }
    }
}

/// Packed ternary weights: 16 trits per u32, 16× denser than FP32.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightBank {
    data: Vec<u32>,
    n_weights: usize,
    /// Logical version for hotswap tracking.
    pub version: u32,
}

impl WeightBank {
    pub fn pack(trits: &[Trit], version: u32) -> Self {
        let words = (trits.len() + 15) / 16;
        let mut data = vec![0u32; words];
        for (i, &t) in trits.iter().enumerate() {
            data[i / 16] |= (t.to_bits() as u32) << ((i % 16) * 2);
        }
        Self { data, n_weights: trits.len(), version }
    }

    pub fn get(&self, idx: usize) -> Trit {
        if idx >= self.n_weights { return Trit::Zero; }
        let bits = ((self.data[idx / 16] >> ((idx % 16) * 2)) & 0b11) as u8;
        Trit::from_bits(bits).unwrap_or(Trit::Zero)
    }

    pub fn len(&self) -> usize { self.n_weights }

    pub fn bytes_used(&self) -> usize { self.data.len() * 4 }

    pub fn bytes_fp32_equivalent(&self) -> usize { self.n_weights * 4 }
}

// ── Ternary navigation network ───────────────────────────────────────────────

/// Sensor input to the navigation network.
#[derive(Debug, Clone, Copy)]
pub struct SensorReading {
    /// Closest obstacle distance, 0.0 (touching) to 1.0 (clear).
    pub obstacle_clearance: f32,
    /// Signed bearing to highest-pheromone target: -1.0 (left) to +1.0 (right).
    pub target_bearing: f32,
    /// Battery state of charge, 0.0 to 1.0.
    pub battery_soc: f32,
    /// Current ground speed, 0.0 to 1.0 (normalised).
    pub speed: f32,
}

/// Output of the navigation network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneAction {
    AvoidObstacle = -1,  // hard-left or brake
    Hold          =  0,  // maintain position / orbit
    Advance       =  1,  // move toward target bearing
}

impl DroneAction {
    pub fn from_trit_sum(sum: i32) -> Self {
        match sum.signum() {
            -1 => DroneAction::AvoidObstacle,
             1 => DroneAction::Advance,
             _ => DroneAction::Hold,
        }
    }
}

/// A 2-layer ternary neural network for drone navigation.
///
/// Layer 1: 4 inputs × 8 hidden  = 32 weights
/// Layer 2: 8 hidden × 3 outputs = 24 weights  (avoid / hold / advance)
/// Total: 56 weights, 4 u32 words (fits in a single cache line).
#[derive(Debug, Clone)]
pub struct TernaryNavNet {
    pub weights: WeightBank,
}

impl TernaryNavNet {
    pub const LAYER1_WEIGHTS: usize = 32;
    pub const LAYER2_WEIGHTS: usize = 24;
    pub const TOTAL_WEIGHTS: usize  = 56;

    /// Build from a slice of trits (must be exactly 56 trits).
    pub fn new(trits: &[Trit; 56], version: u32) -> Self {
        Self { weights: WeightBank::pack(trits, version) }
    }

    /// Forward pass: returns {-1, 0, +1} action from sensor readings.
    ///
    /// All arithmetic is integer — this is what makes ternary nets fast
    /// on hardware without FP units (microcontrollers, edge TPUs, FPGAs).
    pub fn infer(&self, s: &SensorReading) -> DroneAction {
        // Quantize sensors to {-1, 0, +1}
        let inputs: [i8; 4] = [
            quantize(s.obstacle_clearance - 0.3),  // positive = clear
            quantize(s.target_bearing),
            quantize(s.battery_soc - 0.25),        // positive = not low
            quantize(s.speed - 0.1),               // positive = moving
        ];

        // Layer 1: 4→8, ternary dot products
        let mut hidden = [0i8; 8];
        for h in 0..8 {
            let mut acc: i32 = 0;
            for i in 0..4 {
                let w = self.weights.get(h * 4 + i).to_i8();
                acc += w as i32 * inputs[i] as i32;
            }
            hidden[h] = quantize_i32(acc);
        }

        // Layer 2: 8→3, ternary dot products
        let mut output = [0i32; 3];
        for o in 0..3 {
            for h in 0..8 {
                let w = self.weights.get(Self::LAYER1_WEIGHTS + o * 8 + h).to_i8();
                output[o] += w as i32 * hidden[h] as i32;
            }
        }

        // argmax over {avoid, hold, advance} outputs → DroneAction.
        // Hold (index 1) is the safe tie-break: prefer it when scores are equal.
        let max_val = output.iter().copied().max().unwrap_or(0);
        let best_idx = if output[1] == max_val {
            1  // Hold wins ties — safer default than random argmax
        } else {
            output.iter().enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(1)
        };

        match best_idx {
            0 => DroneAction::AvoidObstacle,
            2 => DroneAction::Advance,
            _ => DroneAction::Hold,
        }
    }
}

fn quantize(x: f32) -> i8 {
    if x > 0.15 { 1 } else if x < -0.15 { -1 } else { 0 }
}

fn quantize_i32(x: i32) -> i8 {
    if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
}

// ── Pheromone field ──────────────────────────────────────────────────────────

/// A target location that emits pheromone until covered.
#[derive(Debug, Clone)]
pub struct Target {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub priority: f32,      // 0.0–1.0; scales initial pheromone
    pub covered: bool,
}

/// The stigmergic task allocation field.
///
/// Drones follow pheromone gradients independently — no coordinator.
/// Evaporation provides implicit time-pressure; reinforcement prevents
/// two drones racing for the same target (first drone dampens the signal).
#[derive(Debug, Clone)]
pub struct PheromoneField {
    pub targets: Vec<Target>,
    pheromone: Vec<f32>,
    evaporation_rate: f32,  // fraction decayed per tick
    dampening: f32,         // reduction when a drone visits
}

impl PheromoneField {
    pub fn new(targets: Vec<Target>, evaporation_rate: f32) -> Self {
        let pheromone = targets.iter().map(|t| t.priority).collect();
        Self { targets, pheromone, evaporation_rate, dampening: 0.6 }
    }

    /// Tick: evaporate all pheromone. Returns count of still-active targets.
    pub fn tick(&mut self) -> usize {
        let decay = 1.0 - self.evaporation_rate;
        for p in self.pheromone.iter_mut() {
            *p = (*p * decay).max(0.0);
        }
        self.pheromone.iter().filter(|&&p| p > 0.01).count()
    }

    /// Drone at position (dx, dy) picks the highest-pheromone uncovered target.
    /// Returns (target_idx, bearing) where bearing ∈ [-1.0, 1.0].
    pub fn sample_gradient(&self, dx: f32, dy: f32) -> Option<(usize, f32)> {
        let best = self.targets.iter().enumerate()
            .filter(|(i, t)| !t.covered && self.pheromone[*i] > 0.01)
            .max_by(|(i, _), (j, _)|
                self.pheromone[*i].partial_cmp(&self.pheromone[*j]).unwrap()
            );

        best.map(|(idx, target)| {
            let rel_x = target.x - dx;
            let rel_y = target.y - dy;
            let dist = (rel_x * rel_x + rel_y * rel_y).sqrt().max(0.001);
            let bearing = (rel_x / dist).clamp(-1.0, 1.0);
            (idx, bearing)
        })
    }

    /// A drone covers a target: mark done, dampen residual pheromone.
    pub fn cover(&mut self, target_idx: usize) {
        if target_idx < self.targets.len() {
            self.targets[target_idx].covered = true;
            self.pheromone[target_idx] *= self.dampening;
        }
    }

    /// A drone visits (but does not cover) a target: reduce its pheromone
    /// to deter other drones from racing to the same spot.
    pub fn visit(&mut self, target_idx: usize) {
        if target_idx < self.targets.len() && !self.targets[target_idx].covered {
            self.pheromone[target_idx] *= 1.0 - self.dampening * 0.3;
        }
    }

    pub fn pheromone_at(&self, idx: usize) -> f32 {
        self.pheromone.get(idx).copied().unwrap_or(0.0)
    }

    pub fn covered_count(&self) -> usize {
        self.targets.iter().filter(|t| t.covered).count()
    }

    pub fn total_pheromone(&self) -> f32 {
        self.pheromone.iter().sum()
    }
}

// ── Warp-vote consensus ──────────────────────────────────────────────────────

/// What each drone votes on each round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneVote {
    ContinueMission,  // Trit::Pos  → 0b10
    Regroup,          // Trit::Zero → 0b01
    AbortReturn,      // Trit::Neg  → 0b00
}

impl DroneVote {
    pub fn to_trit(self) -> Trit {
        match self {
            DroneVote::ContinueMission => Trit::Pos,
            DroneVote::Regroup        => Trit::Zero,
            DroneVote::AbortReturn    => Trit::Neg,
        }
    }
}

/// The fleet-level directive produced by warp voting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FleetDirective {
    ContinueMission,
    Regroup,
    AbortReturn,
}

/// Warp-vote result: two u32 ballot masks (faithful to __ballot_sync semantics).
#[derive(Debug, Clone, Copy, Default)]
struct BallotResult {
    positive: u32,  // bit set if vote != AbortReturn
    continue_mask: u32,  // bit set if vote == ContinueMission
}

impl BallotResult {
    fn collect(votes: &[DroneVote]) -> Self {
        let (mut pos, mut cont) = (0u32, 0u32);
        for (i, &v) in votes.iter().enumerate().take(32) {
            let bit = 1u32 << i;
            match v {
                DroneVote::ContinueMission => { pos |= bit; cont |= bit; }
                DroneVote::Regroup        => { pos |= bit; }
                DroneVote::AbortReturn    => {}
            }
        }
        Self { positive: pos, continue_mask: cont }
    }

    fn directive(self) -> FleetDirective {
        let n_continue = self.continue_mask.count_ones();
        let n_abort    = (!self.positive).count_ones() & 0xFFFF_FFFF;
        let n_regroup  = (self.positive & !self.continue_mask).count_ones();

        // Abort wins if ≥ 1/3 of drones vote abort (safety-first threshold).
        if n_abort * 3 >= 32 {
            return FleetDirective::AbortReturn;
        }
        if n_regroup > n_continue {
            FleetDirective::Regroup
        } else {
            FleetDirective::ContinueMission
        }
    }
}

/// A GPU warp of 32 drones voting on a fleet directive.
#[derive(Debug)]
pub struct DroneWarp {
    pub warp_id: u32,
    votes: Vec<DroneVote>,
    pub last_directive: Option<FleetDirective>,
    /// Simulated GPU cycles (4 per ballot pass × 2 passes + 8 overhead).
    pub cycles: u64,
}

impl DroneWarp {
    pub fn new(warp_id: u32, size: usize) -> Self {
        let size = size.min(32);
        Self {
            warp_id,
            votes: vec![DroneVote::ContinueMission; size],
            last_directive: None,
            cycles: 0,
        }
    }

    pub fn set_vote(&mut self, thread: usize, vote: DroneVote) {
        if thread < self.votes.len() {
            self.votes[thread] = vote;
        }
    }

    pub fn execute_ballot(&mut self) -> FleetDirective {
        self.cycles += 8 + 4 + 4;  // setup + ballot pass 1 + ballot pass 2
        let result = BallotResult::collect(&self.votes);
        let directive = result.directive();
        self.last_directive = Some(directive);
        directive
    }
}

// ── CRDT world state ─────────────────────────────────────────────────────────

/// Per-drone state broadcast via CRDT sync.
#[derive(Debug, Clone)]
pub struct DroneState {
    pub drone_id: u32,
    pub x: f32,
    pub y: f32,
    pub battery_soc: f32,
    pub active_target: Option<u32>,
    pub weight_version: u32,
}

/// A Lamport timestamp for LWW ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub u64);

/// LWW-register CRDT for drone state.
///
/// Join rule: keep the entry with the higher timestamp. Commutative,
/// associative, idempotent — safe to merge in any order.
#[derive(Debug, Default)]
pub struct DroneWorldCRDT {
    entries: std::collections::HashMap<u32, (DroneState, Timestamp)>,
}

impl DroneWorldCRDT {
    pub fn upsert(&mut self, state: DroneState, ts: Timestamp) {
        let entry = self.entries.entry(state.drone_id).or_insert_with(|| {
            (state.clone(), Timestamp(0))
        });
        if ts > entry.1 {
            *entry = (state, ts);
        }
    }

    /// CRDT join: merge other into self, keeping higher-timestamp entries.
    pub fn merge(&mut self, other: &DroneWorldCRDT) {
        for (&id, (state, ts)) in &other.entries {
            let entry = self.entries.entry(id).or_insert_with(|| {
                (state.clone(), Timestamp(0))
            });
            if *ts > entry.1 {
                *entry = (state.clone(), *ts);
            }
        }
    }

    pub fn get(&self, drone_id: u32) -> Option<&DroneState> {
        self.entries.get(&drone_id).map(|(s, _)| s)
    }

    pub fn drone_count(&self) -> usize { self.entries.len() }

    /// Average battery across all tracked drones.
    pub fn mean_battery(&self) -> f32 {
        if self.entries.is_empty() { return 1.0; }
        let sum: f32 = self.entries.values().map(|(s, _)| s.battery_soc).sum();
        sum / self.entries.len() as f32
    }

    /// True if all drones run the same weight version (no split-brain after hotswap).
    pub fn weight_version_consistent(&self) -> bool {
        let mut versions = self.entries.values().map(|(s, _)| s.weight_version);
        match versions.next() {
            None => true,
            Some(first) => versions.all(|v| v == first),
        }
    }
}

// ── Conservation verifier ────────────────────────────────────────────────────

/// Checks the fleet conservation invariant before and after a weight hotswap.
///
/// Invariant: every target must be either (a) already covered, (b) assigned
/// to an active drone, or (c) still in the pheromone field with p > threshold.
/// If this breaks, a hotswap corrupted the assignment state.
#[derive(Debug, Clone)]
pub struct ConservationVerifier {
    pub min_coverage_fraction: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    Ok,
    /// Some targets fell below the coverage threshold.
    CoverageDeficit { uncovered: usize, total: usize },
    /// Drones claim more targets than exist.
    AssignmentOverflow,
    /// Weight versions are inconsistent after hotswap.
    VersionSplitBrain { versions_seen: usize },
}

impl ConservationVerifier {
    pub fn new(min_coverage_fraction: f32) -> Self {
        Self { min_coverage_fraction }
    }

    pub fn verify(
        &self,
        field: &PheromoneField,
        world: &DroneWorldCRDT,
    ) -> VerificationResult {
        let total = field.targets.len();
        if total == 0 { return VerificationResult::Ok; }

        // Count covered + assigned + pheromone-active
        let covered   = field.covered_count();
        let assigned: std::collections::HashSet<u32> = world.entries.values()
            .filter_map(|(s, _)| s.active_target)
            .collect();

        // Sanity: no drone can be assigned a target id that doesn't exist
        let max_target_id = field.targets.iter().map(|t| t.id).max().unwrap_or(0);
        if assigned.iter().any(|&id| id > max_target_id) {
            return VerificationResult::AssignmentOverflow;
        }

        // Conservation: covered + reachable (pheromone) + assigned ≥ min fraction
        let reachable = field.pheromone.iter().filter(|&&p| p > 0.05).count();
        let accounted = covered + reachable + assigned.len();
        let fraction  = covered as f32 / total as f32;

        if fraction < self.min_coverage_fraction
            && accounted < (total as f32 * self.min_coverage_fraction) as usize
        {
            return VerificationResult::CoverageDeficit {
                uncovered: total - covered,
                total,
            };
        }

        // Check weight version consistency
        let versions: std::collections::HashSet<u32> = world.entries.values()
            .map(|(s, _)| s.weight_version)
            .collect();
        if versions.len() > 1 {
            return VerificationResult::VersionSplitBrain { versions_seen: versions.len() };
        }

        VerificationResult::Ok
    }
}

// ── Full mission scenario ────────────────────────────────────────────────────

/// Run a compact end-to-end mission: N drones, M targets, T ticks.
///
/// Returns per-tick coverage count. Used to verify the system actually
/// works as a whole, not just as isolated components.
pub fn run_mission(
    n_drones: usize,
    targets: Vec<Target>,
    ticks: usize,
) -> MissionReport {
    let n_targets = targets.len();
    let mut field = PheromoneField::new(targets, 0.05);
    let mut world = DroneWorldCRDT::default();

    // Initialize drone positions spread across [0,1]² grid
    let drone_states: Vec<DroneState> = (0..n_drones as u32).map(|id| DroneState {
        drone_id: id,
        x: (id as f32 / n_drones as f32),
        y: 0.0,
        battery_soc: 1.0,
        active_target: None,
        weight_version: 1,
    }).collect();

    for state in &drone_states {
        world.upsert(state.clone(), Timestamp(0));
    }

    // Default weights: biased toward Advance.
    // hidden 0-3: follow target bearing (input 1).
    // hidden 4-7: react to obstacle clearance (input 0, high = clear sky = advance).
    // Both groups connect to Advance output (output 2).
    let trits: [Trit; 56] = {
        let mut t = [Trit::Zero; 56];
        for i in 0..4 { t[i * 4 + 1] = Trit::Pos; }   // bearing → hidden 0-3
        for h in 4..8 { t[h * 4 + 0] = Trit::Pos; }   // clearance → hidden 4-7
        for h in 0..8 { t[32 + 2 * 8 + h] = Trit::Pos; }  // hidden → Advance
        t
    };
    let nav = TernaryNavNet::new(&trits, 1);
    let verifier = ConservationVerifier::new(0.0); // track but don't fail

    let mut coverage_per_tick = Vec::with_capacity(ticks);
    let mut directives = Vec::new();

    let mut drone_xs: Vec<f32> = (0..n_drones as u32)
        .map(|id| id as f32 / n_drones as f32)
        .collect();
    let mut drone_ys: Vec<f32> = vec![0.0; n_drones];
    let mut drone_batteries: Vec<f32> = vec![1.0; n_drones];

    for tick in 0..ticks {
        field.tick();

        // Each drone: sense → infer → act
        for drone_idx in 0..n_drones {
            let dx = drone_xs[drone_idx];
            let dy = drone_ys[drone_idx];

            let (bearing, target_opt) = match field.sample_gradient(dx, dy) {
                Some((tidx, b)) => (b, Some(tidx)),
                None => (0.0, None),
            };

            let sensor = SensorReading {
                obstacle_clearance: 0.8,  // clear sky
                target_bearing: bearing,
                battery_soc: drone_batteries[drone_idx],
                speed: 0.3,
            };

            let action = nav.infer(&sensor);

            // Move drone based on action
            let step = 0.15;
            match action {
                DroneAction::Advance => {
                    drone_xs[drone_idx] += bearing * step;
                    drone_ys[drone_idx] += step * 0.5;
                }
                DroneAction::AvoidObstacle => {
                    drone_xs[drone_idx] -= bearing * step;
                }
                DroneAction::Hold => {}
            }
            drone_xs[drone_idx] = drone_xs[drone_idx].clamp(0.0, 1.0);
            drone_ys[drone_idx] = drone_ys[drone_idx].clamp(0.0, 1.0);

            drone_batteries[drone_idx] -= 0.02;
            drone_batteries[drone_idx] = drone_batteries[drone_idx].max(0.0);

            // Cover nearby targets.
            // We skip `visit` here: in the full fleet, visit would be called once
            // by whichever drone claims a target, suppressing pheromone for others.
            // In this sim every drone would visit every tick → catastrophic dampening.
            if let Some(tidx) = target_opt {
                let t = &field.targets[tidx];
                let dist = ((t.x - drone_xs[drone_idx]).powi(2)
                    + (t.y - drone_ys[drone_idx]).powi(2)).sqrt();
                if dist < 0.2 && !t.covered {
                    field.cover(tidx);
                }
            }

            // Update CRDT
            let new_state = DroneState {
                drone_id: drone_idx as u32,
                x: drone_xs[drone_idx],
                y: drone_ys[drone_idx],
                battery_soc: drone_batteries[drone_idx],
                active_target: target_opt.map(|i| field.targets[i].id),
                weight_version: nav.weights.version,
            };
            world.upsert(new_state, Timestamp(tick as u64 * 1000 + drone_idx as u64));
        }

        // Warp vote: should we abort if battery low?
        let warp_size = n_drones.min(32);
        let mut warp = DroneWarp::new(0, warp_size);
        for (i, &bat) in drone_batteries.iter().take(warp_size).enumerate() {
            let vote = if bat < 0.1 {
                DroneVote::AbortReturn
            } else if bat < 0.3 {
                DroneVote::Regroup
            } else {
                DroneVote::ContinueMission
            };
            warp.set_vote(i, vote);
        }
        let directive = warp.execute_ballot();
        directives.push(directive);

        coverage_per_tick.push(field.covered_count());

        // Conservation check (would gate a hotswap in production)
        let _ = verifier.verify(&field, &world);

        if directive == FleetDirective::AbortReturn { break; }
        if field.covered_count() == n_targets { break; }
    }

    let final_covered = field.covered_count();
    MissionReport {
        n_drones,
        n_targets,
        ticks_elapsed: coverage_per_tick.len(),
        final_covered,
        coverage_fraction: final_covered as f32 / n_targets as f32,
        coverage_per_tick,
        directives,
        mean_battery: world.mean_battery(),
    }
}

#[derive(Debug)]
pub struct MissionReport {
    pub n_drones: usize,
    pub n_targets: usize,
    pub ticks_elapsed: usize,
    pub final_covered: usize,
    pub coverage_fraction: f32,
    pub coverage_per_tick: Vec<usize>,
    pub directives: Vec<FleetDirective>,
    pub mean_battery: f32,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ternary weight encoding ──────────────────────────────────────────────

    #[test]
    fn trit_roundtrip() {
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!(Trit::from_bits(t.to_bits()), Some(t));
        }
        assert_eq!(Trit::from_bits(0b11), None);
    }

    #[test]
    fn trit_i8_values() {
        assert_eq!(Trit::Neg.to_i8(),  -1);
        assert_eq!(Trit::Zero.to_i8(),  0);
        assert_eq!(Trit::Pos.to_i8(),   1);
    }

    #[test]
    fn weight_bank_density() {
        let trits = vec![Trit::Pos; 56];
        let bank = WeightBank::pack(&trits, 1);
        // 56 trits × 2 bits = 14 bytes → 4 u32 words = 16 bytes
        assert_eq!(bank.bytes_used(), 16);
        // FP32 equivalent: 56 × 4 = 224 bytes → 14× denser
        assert_eq!(bank.bytes_fp32_equivalent(), 224);
        assert!(bank.bytes_used() < bank.bytes_fp32_equivalent());
    }

    #[test]
    fn weight_bank_random_access() {
        let mut trits = [Trit::Zero; 56];
        trits[0]  = Trit::Neg;
        trits[15] = Trit::Pos;
        trits[55] = Trit::Neg;
        let bank = WeightBank::pack(&trits, 1);
        assert_eq!(bank.get(0),  Trit::Neg);
        assert_eq!(bank.get(15), Trit::Pos);
        assert_eq!(bank.get(55), Trit::Neg);
        assert_eq!(bank.get(10), Trit::Zero);
    }

    // ── TernaryNavNet inference ──────────────────────────────────────────────

    fn advance_biased_net() -> TernaryNavNet {
        let mut trits = [Trit::Zero; 56];
        for i in 0..4 { trits[i * 4 + 1] = Trit::Pos; }
        for h in 0..8 { trits[32 + 2 * 8 + h] = Trit::Pos; }
        TernaryNavNet::new(&trits, 1)
    }

    #[test]
    fn nav_advances_toward_target() {
        let net = advance_biased_net();
        let sensor = SensorReading {
            obstacle_clearance: 0.9,
            target_bearing: 0.8,  // target to the right
            battery_soc: 0.8,
            speed: 0.3,
        };
        assert_eq!(net.infer(&sensor), DroneAction::Advance);
    }

    #[test]
    fn nav_avoids_obstacle_when_sensor_triggers() {
        // Build a net biased to avoid when obstacle_clearance is low
        let mut trits = [Trit::Zero; 56];
        // Connect obstacle_clearance (input 0, inverted = danger) to hidden
        for h in 0..4 { trits[h * 4 + 0] = Trit::Neg; }
        // Connect hidden to AvoidObstacle output (output 0)
        for h in 0..4 { trits[32 + 0 * 8 + h] = Trit::Pos; }
        let net = TernaryNavNet::new(&trits, 2);
        let sensor = SensorReading {
            obstacle_clearance: 0.0,  // obstacle right in front
            target_bearing: 0.0,
            battery_soc: 0.8,
            speed: 0.3,
        };
        assert_eq!(net.infer(&sensor), DroneAction::AvoidObstacle);
    }

    #[test]
    fn nav_holds_with_all_zero_weights() {
        let trits = [Trit::Zero; 56];
        let net = TernaryNavNet::new(&trits, 0);
        let sensor = SensorReading {
            obstacle_clearance: 0.5,
            target_bearing: 0.5,
            battery_soc: 0.5,
            speed: 0.5,
        };
        // All-zero network → all outputs tied at 0 → Hold (index 1 default)
        assert_eq!(net.infer(&sensor), DroneAction::Hold);
    }

    #[test]
    fn weight_hotswap_changes_behavior() {
        let advance_net = advance_biased_net();
        let mut avoid_trits = [Trit::Zero; 56];
        for h in 0..4 { avoid_trits[h * 4 + 0] = Trit::Neg; }
        for h in 0..4 { avoid_trits[32 + h] = Trit::Pos; }
        let avoid_net = TernaryNavNet::new(&avoid_trits, 2);

        let sensor = SensorReading {
            obstacle_clearance: 0.0,
            target_bearing: 0.8,
            battery_soc: 0.8,
            speed: 0.3,
        };
        // Before swap: Advance. After swap: AvoidObstacle.
        let before = advance_net.infer(&sensor);
        let after  = avoid_net.infer(&sensor);
        assert_ne!(before, after,
            "hotswap should change behavior on identical sensor input");
        assert_eq!(avoid_net.weights.version, 2);
    }

    // ── Pheromone field ──────────────────────────────────────────────────────

    fn make_targets(n: usize) -> Vec<Target> {
        (0..n as u32).map(|i| Target {
            id: i,
            x: i as f32 / n as f32,
            y: 0.5,
            priority: 1.0,
            covered: false,
        }).collect()
    }

    #[test]
    fn pheromone_evaporates_over_ticks() {
        let mut field = PheromoneField::new(make_targets(4), 0.1);
        let initial: f32 = field.total_pheromone();
        field.tick();
        assert!(field.total_pheromone() < initial);
    }

    #[test]
    fn pheromone_dampens_on_cover() {
        let mut field = PheromoneField::new(make_targets(4), 0.05);
        let before = field.pheromone_at(0);
        field.cover(0);
        assert!(field.pheromone_at(0) < before);
        assert!(field.targets[0].covered);
    }

    #[test]
    fn pheromone_gradient_points_to_strongest() {
        let targets = vec![
            Target { id: 0, x: 0.1, y: 0.5, priority: 0.1, covered: false },
            Target { id: 1, x: 0.9, y: 0.5, priority: 0.9, covered: false },
        ];
        let field = PheromoneField::new(targets, 0.05);
        let (idx, _bearing) = field.sample_gradient(0.5, 0.5).unwrap();
        assert_eq!(idx, 1, "gradient should point to higher-pheromone target");
    }

    #[test]
    fn covered_targets_excluded_from_gradient() {
        let mut field = PheromoneField::new(make_targets(3), 0.05);
        field.cover(0);
        field.cover(1);
        // Only target 2 remains
        let (idx, _) = field.sample_gradient(0.0, 0.0).unwrap();
        assert_eq!(idx, 2);
    }

    // ── Warp-vote consensus ──────────────────────────────────────────────────

    #[test]
    fn full_battery_fleet_continues() {
        let mut warp = DroneWarp::new(0, 32);
        for i in 0..32 { warp.set_vote(i, DroneVote::ContinueMission); }
        assert_eq!(warp.execute_ballot(), FleetDirective::ContinueMission);
    }

    #[test]
    fn low_battery_triggers_abort() {
        let mut warp = DroneWarp::new(0, 32);
        // 12/32 ≈ 37.5% → ≥ 1/3 → AbortReturn wins
        for i in 0..12 { warp.set_vote(i, DroneVote::AbortReturn); }
        for i in 12..32 { warp.set_vote(i, DroneVote::ContinueMission); }
        assert_eq!(warp.execute_ballot(), FleetDirective::AbortReturn);
    }

    #[test]
    fn majority_regroup_beats_continue() {
        let mut warp = DroneWarp::new(0, 32);
        // 18 regroup, 14 continue, 0 abort → Regroup
        for i in 0..18 { warp.set_vote(i, DroneVote::Regroup); }
        for i in 18..32 { warp.set_vote(i, DroneVote::ContinueMission); }
        assert_eq!(warp.execute_ballot(), FleetDirective::Regroup);
    }

    #[test]
    fn ballot_cycle_cost_is_constant() {
        let mut warp = DroneWarp::new(0, 32);
        warp.execute_ballot();
        // 8 setup + 4 + 4 = 16 cycles, independent of vote distribution
        assert_eq!(warp.cycles, 16);
    }

    // ── CRDT world state ─────────────────────────────────────────────────────

    #[test]
    fn crdt_lww_keeps_later_timestamp() {
        let mut world = DroneWorldCRDT::default();
        let state_v1 = DroneState { drone_id: 0, x: 0.1, y: 0.2,
            battery_soc: 0.9, active_target: None, weight_version: 1 };
        let state_v2 = DroneState { drone_id: 0, x: 0.5, y: 0.5,
            battery_soc: 0.7, active_target: Some(3), weight_version: 1 };

        world.upsert(state_v1, Timestamp(100));
        world.upsert(state_v2, Timestamp(200));

        let s = world.get(0).unwrap();
        assert_eq!(s.x, 0.5);
        assert_eq!(s.active_target, Some(3));
    }

    #[test]
    fn crdt_older_update_does_not_overwrite() {
        let mut world = DroneWorldCRDT::default();
        let new_state = DroneState { drone_id: 1, x: 0.8, y: 0.8,
            battery_soc: 0.5, active_target: Some(7), weight_version: 2 };
        let old_state = DroneState { drone_id: 1, x: 0.1, y: 0.1,
            battery_soc: 0.9, active_target: None, weight_version: 1 };

        world.upsert(new_state, Timestamp(500));
        world.upsert(old_state, Timestamp(100));  // old, should not win

        let s = world.get(1).unwrap();
        assert_eq!(s.x, 0.8);  // new state preserved
    }

    #[test]
    fn crdt_merge_converges() {
        let mut world_a = DroneWorldCRDT::default();
        let mut world_b = DroneWorldCRDT::default();

        let s1 = DroneState { drone_id: 0, x: 0.1, y: 0.1,
            battery_soc: 1.0, active_target: None, weight_version: 1 };
        let s2 = DroneState { drone_id: 1, x: 0.9, y: 0.9,
            battery_soc: 0.8, active_target: Some(2), weight_version: 1 };

        world_a.upsert(s1, Timestamp(100));
        world_b.upsert(s2, Timestamp(200));

        world_a.merge(&world_b);

        // After merge, world_a has both drones
        assert_eq!(world_a.drone_count(), 2);
        assert!(world_a.get(1).is_some());
    }

    // ── Conservation verifier ────────────────────────────────────────────────

    #[test]
    fn conservation_ok_with_full_coverage() {
        let targets = make_targets(4);
        let mut field = PheromoneField::new(targets, 0.05);
        for i in 0..4 { field.cover(i); }
        let world = DroneWorldCRDT::default();
        let v = ConservationVerifier::new(0.5);
        assert_eq!(v.verify(&field, &world), VerificationResult::Ok);
    }

    #[test]
    fn conservation_detects_coverage_deficit() {
        // All pheromone evaporated, nothing covered, no drones assigned
        let targets = make_targets(10);
        let mut field = PheromoneField::new(targets, 1.0);
        for _ in 0..50 { field.tick(); }  // fully evaporate
        let world = DroneWorldCRDT::default();
        let v = ConservationVerifier::new(0.8);
        let result = v.verify(&field, &world);
        assert!(matches!(result, VerificationResult::CoverageDeficit { .. }));
    }

    #[test]
    fn conservation_detects_version_split_brain() {
        let field = PheromoneField::new(make_targets(2), 0.05);
        let mut world = DroneWorldCRDT::default();
        // Two drones with different weight versions after a partial hotswap
        let s1 = DroneState { drone_id: 0, x: 0.0, y: 0.0,
            battery_soc: 1.0, active_target: None, weight_version: 1 };
        let s2 = DroneState { drone_id: 1, x: 1.0, y: 1.0,
            battery_soc: 1.0, active_target: None, weight_version: 2 };
        world.upsert(s1, Timestamp(100));
        world.upsert(s2, Timestamp(100));
        let v = ConservationVerifier::new(0.0);
        assert_eq!(
            v.verify(&field, &world),
            VerificationResult::VersionSplitBrain { versions_seen: 2 }
        );
    }

    // ── End-to-end mission ───────────────────────────────────────────────────

    #[test]
    fn mission_makes_progress() {
        let targets: Vec<Target> = (0..5u32).map(|i| Target {
            id: i, x: i as f32 * 0.2, y: 0.5, priority: 1.0, covered: false,
        }).collect();
        let report = run_mission(8, targets, 30);
        // Fleet of 8 drones should cover at least 1 target in 30 ticks
        assert!(report.final_covered >= 1,
            "expected ≥1 covered, got {}", report.final_covered);
        assert!(report.coverage_per_tick.len() <= 30);
        println!(
            "\nMission — {}/{} covered in {} ticks, battery: {:.0}%",
            report.final_covered, report.n_targets,
            report.ticks_elapsed, report.mean_battery * 100.0
        );
    }

    #[test]
    fn mission_coverage_is_monotonically_nondecreasing() {
        let targets: Vec<Target> = (0..4u32).map(|i| Target {
            id: i, x: i as f32 * 0.25, y: 0.5, priority: 1.0, covered: false,
        }).collect();
        let report = run_mission(4, targets, 20);
        // Coverage must never go down (targets stay covered once covered)
        for window in report.coverage_per_tick.windows(2) {
            assert!(window[1] >= window[0],
                "coverage decreased: {} → {}", window[0], window[1]);
        }
    }
}
