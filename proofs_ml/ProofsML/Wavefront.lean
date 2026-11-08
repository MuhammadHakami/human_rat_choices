/-
  Vision4Neuro stage 7 (2026-11-07): a hippocampal goal map transfers to never-seen rooms because following it cannot trap
  the agent. Place cells S (finite), goal memory G ≥ 0 (dopamine-gated Hebbian trace at rewarded places), recurrent
  wavefront value v x = max (G x) (γ · max over believed-open neighbours v y), 0 < γ < 1 (Ponulak & Hopfield 2013).
  1. wave_ascent: at any cell that is not a goal source (G x < v x) some open neighbour has strictly larger value:
     the wavefront has no local maximum away from a remembered goal, in ANY layout (no view or room term appears).
  2. greedy_reaches_source: a step rule that always climbs v where possible reaches a cell with v = G (a goal source)
     within |S| steps. A bump only removes a believed-open link, so the same holds on the repaired map.
  Standard axioms only.
-/
import Mathlib

theorem wave_ascent {S : Type*} (v G : S → ℝ) (N : S → Set S) (γ : ℝ) (hγ0 : 0 < γ) (hγ1 : γ < 1)
    (hG : ∀ x, 0 ≤ G x) (hrec : ∀ x, G x < v x → ∃ y ∈ N x, v x = γ * v y) (x : S) (hx : G x < v x) :
    ∃ y ∈ N x, v x < v y := by
  obtain ⟨y, hy, he⟩ := hrec x hx
  refine ⟨y, hy, ?_⟩
  have hv : 0 < v x := lt_of_le_of_lt (hG x) hx
  have hvy : 0 < v y := by
    by_contra h; push Not at h; nlinarith
  nlinarith

theorem greedy_reaches_source {S : Type*} [Fintype S] (v G : S → ℝ) (f : S → S)
    (hstep : ∀ x, G x < v x → v x < v (f x)) (x : S) :
    ∃ n ≤ Fintype.card S, ¬ G (f^[n] x) < v (f^[n] x) := by
  by_contra h
  push Not at h
  have mono : ∀ m n, m < n → n ≤ Fintype.card S → v (f^[m] x) < v (f^[n] x) := by
    intro m n hmn hn
    induction n with
    | zero => omega
    | succ k ih =>
      have hk : G (f^[k] x) < v (f^[k] x) := h k (by omega)
      have step : v (f^[k] x) < v (f^[k + 1] x) := by
        rw [Function.iterate_succ_apply']; exact hstep _ hk
      rcases Nat.lt_succ_iff_lt_or_eq.mp hmn with h' | h'
      · exact lt_trans (ih h' (by omega)) step
      · subst h'; exact step
  have inj : Function.Injective (fun i : Fin (Fintype.card S + 1) => f^[i.val] x) := by
    intro i j hij
    simp only at hij
    by_contra hne
    have hi := Nat.lt_succ_iff.mp i.isLt
    have hj := Nat.lt_succ_iff.mp j.isLt
    rcases lt_or_gt_of_ne (Fin.val_injective.ne hne) with hl | hl
    · have := mono _ _ hl hj; rw [hij] at this; exact lt_irrefl _ this
    · have := mono _ _ hl hi; rw [hij] at this; exact lt_irrefl _ this
  have := Fintype.card_le_of_injective _ inj
  rw [Fintype.card_fin] at this; omega
