/-
  ArcFly cost model — machine-checked.

  The paper claims the spiking mushroom-body learner uses "~19x fewer operations per decision"
  and "~40x fewer learned parameters" than the PPO baseline. Those are arithmetic consequences
  of the two architectures, not measurements, so they can be proved rather than asserted.

  Every constant below is a value the running code actually reports:

    arc_fly/mushroom_body.py : n_kc=300, mean fanin=3, k_active=18, N_ODOR=3
                               ops_per_decision() -> {kc_stage:900, spike_readout:54, total:954}
                               n_learned_params   -> 900
    arc_fly/ppo_baseline.py  : SB3 PPO("MlpPolicy", net_arch=[128,128]), obs=11, act=1
                               instantiated model -> 36,355 parameters

  No Mathlib: these are decidable statements about concrete naturals, so `rfl`, `decide` and
  `omega` discharge them and the build stays seconds rather than an hour.
-/

namespace ArcFly

/-! ## 1. Input space and constraints -/

/-- The spiking mushroom-body readout. `h_sparse` pins the k-winners-take-all sparsity at 6%,
which is the claim the sparse-coding / continual-learning argument rests on. -/
structure MBParams where
  n_kc     : Nat
  fanin    : Nat   -- mean nonzero context->KC weights per Kenyon cell
  k_active : Nat   -- Kenyon cells surviving k-WTA
  n_odor   : Nat   -- MBON output channels
  h_kc_pos : 0 < n_kc
  h_sparse : k_active * 100 = 6 * n_kc

/-- A Stable-Baselines3 `MlpPolicy` with two hidden layers of equal width. -/
structure PPOParams where
  obs : Nat
  hid : Nat
  act : Nat

/-! ## 2. Transfer functions: architecture -> cost -/

/-- Synaptic operations for one decision: a dense context->KC stage plus an event-driven
KC->MBON readout that only touches the `k_active` cells that spiked. -/
def mbOps (p : MBParams) : Nat := p.n_kc * p.fanin + p.k_active * p.n_odor

/-- Plastic weights. Only the KC->MBON readout learns; the connectome stays frozen. -/
def mbLearned (p : MBParams) : Nat := p.n_odor * p.n_kc

/-- Multiply-accumulates on the policy path of one forward pass (weights only, no biases):
obs->hidden, hidden->hidden, hidden->action. -/
def ppoOps (q : PPOParams) : Nat := q.obs * q.hid + q.hid * q.hid + q.hid * q.act

/-- Every trainable scalar SB3 allocates: separate policy and value trunks (each with biases),
an action head, a value head, and the state-independent log-std. -/
def ppoLearned (q : PPOParams) : Nat :=
  2 * ((q.obs * q.hid + q.hid) + (q.hid * q.hid + q.hid))
    + (q.hid * q.act + q.act)
    + (q.hid + 1)
    + q.act

/-! ## 3. The configurations actually used -/

def arcFlyMB : MBParams :=
  { n_kc := 300, fanin := 3, k_active := 18, n_odor := 3,
    h_kc_pos := by decide, h_sparse := by decide }

def arcFlyPPO : PPOParams := { obs := 11, hid := 128, act := 1 }

/-! ## 4. The reported counts are exactly these architectures' costs -/

theorem mb_ops_is_954 : mbOps arcFlyMB = 954 := rfl

theorem mb_kc_stage_is_900 : arcFlyMB.n_kc * arcFlyMB.fanin = 900 := rfl

theorem mb_readout_is_54 : arcFlyMB.k_active * arcFlyMB.n_odor = 54 := rfl

theorem mb_learned_is_900 : mbLearned arcFlyMB = 900 := rfl

theorem ppo_ops_is_17920 : ppoOps arcFlyPPO = 17920 := rfl

theorem ppo_learned_is_36355 : ppoLearned arcFlyPPO = 36355 := rfl

/-! ## 5. The efficiency ratios, bounded from BOTH sides

`~19x` and `~40x` are round numbers. Bounding each ratio above and below shows exactly how
much rounding is going on, so the paper can state the claim without overstating it. -/

/-- Operations: at least 18x fewer. -/
theorem ops_ratio_ge_18 : mbOps arcFlyMB * 18 ≤ ppoOps arcFlyPPO := by decide

/-- Operations: but strictly LESS than 19x. The true ratio is 17920/954 = 18.78..., so
"~19x fewer" is a round-up and ">19x" would be false. -/
theorem ops_ratio_lt_19 : ppoOps arcFlyPPO < mbOps arcFlyMB * 19 := by decide

/-- Parameters: at least 40x fewer. -/
theorem params_ratio_ge_40 : mbLearned arcFlyMB * 40 ≤ ppoLearned arcFlyPPO := by decide

/-- Parameters: and strictly less than 41x. True ratio 36355/900 = 40.39..., so "~40x" is
accurate to the stated precision. -/
theorem params_ratio_lt_41 : ppoLearned arcFlyPPO < mbLearned arcFlyMB * 41 := by decide

/-! ## 6. Sparsity -/

/-- The k-WTA layer keeps exactly 6% of Kenyon cells: 18 of 300. -/
theorem sparsity_is_six_percent : arcFlyMB.k_active * 100 = 6 * arcFlyMB.n_kc :=
  arcFlyMB.h_sparse

/-- Only the sparse survivors drive the plastic readout, so the event-driven cost is a fixed
6% of what a dense KC->MBON readout would cost. -/
theorem readout_is_six_percent_of_dense :
    (arcFlyMB.k_active * arcFlyMB.n_odor) * 100 = 6 * (arcFlyMB.n_kc * arcFlyMB.n_odor) := by
  decide

/-! ## 7. Where the asymmetry actually comes from

Not a coincidence of the chosen constants: PPO's cost is quadratic in hidden width while the
mushroom body's is linear in `n_kc`. These hold for EVERY configuration, not just ours. -/

/-- The hidden-hidden matmul alone lower-bounds PPO's per-decision cost. -/
theorem ppo_ops_ge_hidden_sq (q : PPOParams) : q.hid * q.hid ≤ ppoOps q := by
  unfold ppoOps
  omega

/-- The mushroom body's cost is linear in the number of Kenyon cells. -/
theorem mb_ops_linear (p : MBParams) :
    mbOps p = p.n_kc * p.fanin + p.k_active * p.n_odor := rfl

/-- Consequently, for any PPO whose squared hidden width exceeds the mushroom body's total
cost, the bio learner is strictly cheaper — the advantage is structural, not tuned. -/
theorem bio_cheaper_when_hidden_large (p : MBParams) (q : PPOParams)
    (h : mbOps p < q.hid * q.hid) : mbOps p < ppoOps q :=
  Nat.lt_of_lt_of_le h (ppo_ops_ge_hidden_sq q)

end ArcFly
