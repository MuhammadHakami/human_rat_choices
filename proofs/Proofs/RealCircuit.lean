/-
  Cost of the MEASURED mushroom body.

  `Proofs.Basic` proves the cost model of the paper's 300-KC / 3-MBON abstraction. Those numbers
  are not the fly's. This file states the same claims for the circuit actually extracted from
  FlyWire v783 (checkpoint/connectome/mb_subgraph.npz), so the paper can quote the real figure.

  Edge counts are measured, not chosen:
    PN->KC   27,848      KC->MBON 62,261      KC->APL 5,215      APL->KC 5,194
  and 62,261 of those synapses are the plastic ones.
-/

namespace ArcFly.Real

structure Circuit where
  pnKC   : Nat
  kcMBON : Nat
  kcAPL  : Nat
  aplKC  : Nat

/-- One decision drives PN->KC, the APL feedback loop, and the KC->MBON readout. -/
def opsPerDecision (c : Circuit) : Nat := c.pnKC + c.kcAPL + c.aplKC + c.kcMBON

/-- Only the measured KC->MBON synapses are plastic. -/
def plastic (c : Circuit) : Nat := c.kcMBON

/-- The subgraph extracted from FlyWire v783. -/
def flywire : Circuit := { pnKC := 27848, kcMBON := 62261, kcAPL := 5215, aplKC := 5194 }

theorem ops_is_100518 : opsPerDecision flywire = 100518 := rfl

theorem plastic_is_62261 : plastic flywire = 62261 := rfl

/-- The abstraction in `Proofs.Basic` costs 954 ops; the real circuit costs more than 100x that.
Stating it keeps the paper from quoting the flattering number. -/
theorem real_costs_more_than_100x_abstraction : 954 * 100 < opsPerDecision flywire := by decide

/-! ## The efficiency claim, with the LLM cost BRACKETED rather than guessed

The previous version fixed the harness cost at `2 * (2*10^12) * 500 = 2*10^15` operations per
environment action. Two FLOPs per parameter per generated token is standard (Kaplan et al.
2020; Hoffmann et al. 2022), but BOTH remaining factors were guesses: frontier parameter counts
are undisclosed, and mixture-of-experts models activate far fewer parameters than they hold, so
the true active count may be ~3*10^10 rather than 2*10^12. Tokens per action is likewise a
choice, from ~100 for a terse policy to ~5000 for a long-context recursive harness.

Guessing high flatters us by three orders of magnitude, so we prove the claim at the
CONSERVATIVE end -- smallest plausible model, fewest tokens -- and let the favourable end be
reported only as a range. -/

/-- 2 FLOPs/param/token * 3*10^10 active params * 100 tokens: the cheapest plausible harness. -/
def llmOpsConservative : Nat := 6000000000000

/-- 2 FLOPs/param/token * 2*10^12 params * 500 tokens: the previous, most flattering estimate. -/
def llmOpsAggressive : Nat := 2000000000000000

/-- Even against the CHEAPEST plausible LLM harness, the measured circuit is >10^7 times
cheaper per decision. This is the number the paper is entitled to lead with. -/
theorem real_circuit_cheaper_by_1e7_conservative :
    opsPerDecision flywire * 10000000 < llmOpsConservative := by decide

/-- And strictly below 10^8 at that end, so "seven orders of magnitude" is the honest phrasing
for the conservative case; ten orders is NOT defensible without assuming a large dense model. -/
theorem conservative_ratio_below_1e8 :
    llmOpsConservative < opsPerDecision flywire * 100000000 := by decide

/-- The aggressive end still exceeds 10^10, so the full honest statement is a BAND:
the advantage is between ~10^7 and ~10^10 depending on undisclosed model size and token budget. -/
theorem aggressive_end_exceeds_1e10 :
    opsPerDecision flywire * 10000000000 < llmOpsAggressive := by decide

/-- The two corrections we are obliged to state together: the real circuit costs more than the
abstraction, AND the efficiency claim must be quoted at its conservative floor. -/
theorem correction_direction :
    954 < opsPerDecision flywire ∧
    opsPerDecision flywire * 10000000 < llmOpsConservative := by
  refine ⟨by decide, by decide⟩

end ArcFly.Real
