/-
  Vision4Neuro camera-ready (2026-11-05): the eligibility trace of the brain-constrained circuit is bounded.
  Trace update e' = q e + z with decay q = gamma*lambda in [0,1) and |z| <= M (unit-norm codes: M = 1).
  Written over Int with q = a/b (0 <= a < b) and the step multiplied through by b:  b e' = a e + b z.
  Invariant: (b - a) e <= b M, i.e. e <= M / (1 - q). Hence a value-rate eta_v changes V(s) by at most
  eta_v |delta| M / (1 - q) per step: the search measures eta_v in units of (1 - gamma lambda) (rho = eta_v / (1 - q)).
-/

namespace V4N

/-- One step preserves the bound e <= M/(1-q) (upper side). -/
theorem trace_step_upper (a b e e' z M : Int) (hb : 0 < b) (ha : 0 ≤ a) (hab : a < b)
    (hstep : b * e' = a * e + b * z) (he : (b - a) * e ≤ b * M) (hz : z ≤ M) :
    (b - a) * e' ≤ b * M := by
  have h1 : a * ((b - a) * e) ≤ a * (b * M) := Int.mul_le_mul_of_nonneg_left he ha
  have h2 : (b - a) * (b * z) ≤ (b - a) * (b * M) :=
    Int.mul_le_mul_of_nonneg_left (Int.mul_le_mul_of_nonneg_left hz (Int.le_of_lt hb)) (by omega)
  have key : b * ((b - a) * e') ≤ b * (b * M) := by
    have : b * ((b - a) * e') = a * ((b - a) * e) + (b - a) * (b * z) := by
      rw [show b * ((b - a) * e') = (b - a) * (b * e') by grind, hstep]; grind
    rw [this]; have : a * (b * M) + (b - a) * (b * M) = b * (b * M) := by grind
    omega
  exact Int.le_of_mul_le_mul_left key hb

/-- The same for the lower side (traces of signed inputs): -M/(1-q) <= e. -/
theorem trace_step_lower (a b e e' z M : Int) (hb : 0 < b) (ha : 0 ≤ a) (hab : a < b)
    (hstep : b * e' = a * e + b * z) (he : -(b * M) ≤ (b - a) * e) (hz : -M ≤ z) :
    -(b * M) ≤ (b - a) * e' := by
  have := trace_step_upper a b (-e) (-e') (-z) M hb ha hab (by grind) (by grind) (by omega)
  grind

/-- A trace run from zero over inputs bounded by M stays within M/(1-q) for every length. -/
def traceRun (a b : Int) : List Int → Int → Int → Prop
  | [], e, e_end => e_end = e
  | z :: zs, e, e_end => ∃ e', b * e' = a * e + b * z ∧ traceRun a b zs e' e_end

theorem trace_bounded (a b M : Int) (hb : 0 < b) (ha : 0 ≤ a) (hab : a < b) (zs : List Int) :
    ∀ e e_end, (b - a) * e ≤ b * M → (∀ z ∈ zs, z ≤ M) → traceRun a b zs e e_end → (b - a) * e_end ≤ b * M := by
  induction zs with
  | nil => intro e e_end he _ hr; simp [traceRun] at hr; subst hr; exact he
  | cons z zs ih =>
    intro e e_end he hz hr
    obtain ⟨e', hs, hr'⟩ := hr
    exact ih e' e_end (trace_step_upper a b e e' z M hb ha hab hs he (hz z (by simp)))
      (fun w hw => hz w (by simp [hw])) hr'

/-- The stage-2 winners in these units (gamma = 0.95; rho = eta_v / (1 - gamma lambda), floor in hundredths):
ungated lambda = .898, eta_v = .2655 -> 1 - q = .1469, rho = 1.80; gated lambda = .957, eta_v = .0641 -> .0909, rho = .70. -/
theorem stage2_rho_ungated : (2655 : Int) * 100 / 1469 = 180 := by decide
theorem stage2_rho_gated : (641 : Int) * 100 / 909 = 70 := by decide

end V4N
