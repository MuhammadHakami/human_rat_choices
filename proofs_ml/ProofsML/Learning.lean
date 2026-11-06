/-
  Vision4Neuro camera-ready (2026-10-31): Mathlib proofs for the learning rules and the loop argument.
  1. softmax_logit_grad: d/dz_i log softmax_a(z) = [i = a] - softmax_i(z). With z = W h + b, the TD-readout's local
     eligibility increment (onehot(a) - pi) h^T is therefore exactly the policy gradient of the output synapses.
  2. trace_closed_form: the online trace e_t = (gamma lambda) e_{t-1} + g_t equals sum_k (gamma lambda)^(t-k) g_k.
  3. orbit_repeats: a deterministic orbit on a finite state space repeats within card S steps.
-/
import Mathlib

open Real Finset

namespace V4N

/-! ## 1. Log-softmax gradient -/

/-- Logits perturbed by t along coordinate i. -/
noncomputable def bump {n : ℕ} (z : Fin n → ℝ) (i : Fin n) (t : ℝ) : Fin n → ℝ := fun k => z k + if k = i then t else 0

/-- log softmax_a(z) = z_a - log (sum_k exp z_k). -/
noncomputable def logSoftmax {n : ℕ} (z : Fin n → ℝ) (a : Fin n) : ℝ := z a - Real.log (∑ k, Real.exp (z k))

noncomputable def softmax {n : ℕ} (z : Fin n → ℝ) (i : Fin n) : ℝ := Real.exp (z i) / ∑ k, Real.exp (z k)

theorem softmax_logit_grad {n : ℕ} (z : Fin n → ℝ) (a i : Fin n) :
    HasDerivAt (fun t => logSoftmax (bump z i t) a) ((if a = i then 1 else 0) - softmax z i) 0 := by
  have hpos : 0 < ∑ k, Real.exp (z k) := by
    have : Nonempty (Fin n) := ⟨a⟩
    exact Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty
  -- derivative of each summand exp (z k + [k = i] t)
  have hk : ∀ k ∈ (univ : Finset (Fin n)), HasDerivAt (fun t => Real.exp (z k + if k = i then t else 0))
      (Real.exp (z k) * (if k = i then 1 else 0)) 0 := by
    intro k _
    by_cases h : k = i
    · subst h; simp only [if_true]
      have := ((hasDerivAt_id (0:ℝ)).const_add (z k)).exp; simpa using this
    · simp only [h, if_false, add_zero]; simpa using hasDerivAt_const (0:ℝ) (Real.exp (z k))
  have hsum := HasDerivAt.sum hk
  have hsum' : HasDerivAt (fun t => ∑ k, Real.exp (z k + if k = i then t else 0)) (Real.exp (z i)) 0 := by
    refine (hsum.congr_deriv ?_).congr_of_eventuallyEq (Filter.Eventually.of_forall (fun t => ?_))
    · simp [mul_ite, Finset.sum_ite_eq']
    · simp [Finset.sum_apply]
  have h0 : (∑ k, Real.exp (z k + if k = i then (0:ℝ) else 0)) = ∑ k, Real.exp (z k) := by simp
  have hlog := hsum'.log (by rw [h0]; exact hpos.ne')
  have hza : HasDerivAt (fun t => z a + if a = i then t else 0) (if a = i then 1 else 0) 0 := by
    by_cases h : a = i
    · subst h; simpa using (hasDerivAt_id (0:ℝ)).const_add (z a)
    · simp only [h, if_false, add_zero]; exact hasDerivAt_const _ _
  have key : (fun t => logSoftmax (bump z i t) a) =
      fun t => (z a + if a = i then t else 0) - Real.log (∑ k, Real.exp (z k + if k = i then t else 0)) := by
    funext t; simp [logSoftmax, bump]
  rw [key]
  refine (hza.sub hlog).congr_deriv ?_
  rw [h0]; simp [softmax]

/-! ## 2. Eligibility trace closed form -/

/-- Online trace: e_0 = g_0, e_{t+1} = c * e_t + g_{t+1} (c = gamma * lambda, traces reset at episode start). -/
def trace (c : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => g 0
  | t + 1 => c * trace c g t + g (t + 1)

theorem trace_closed_form (c : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    trace c g t = ∑ k ∈ range (t + 1), c ^ (t - k) * g k := by
  induction t with
  | zero => simp [trace]
  | succ t ih =>
    rw [trace, ih, Finset.sum_range_succ _ (t + 1), Finset.mul_sum]
    congr 1
    · apply Finset.sum_congr rfl; intro k hk
      have : k ≤ t := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      rw [show t + 1 - k = (t - k) + 1 by omega, pow_succ]; ring
    · simp

/-! ## 3. Deterministic orbits on a finite state space repeat within card S steps -/

theorem orbit_repeats {S : Type} [Fintype S] (f : S → S) (s : S) :
    ∃ i j, i < j ∧ j ≤ Fintype.card S ∧ f^[i] s = f^[j] s := by
  have hlt : Fintype.card S < Fintype.card (Fin (Fintype.card S + 1)) := by simp
  obtain ⟨a, b, hab, he⟩ := Fintype.exists_ne_map_eq_of_card_lt (fun k : Fin (Fintype.card S + 1) => f^[k.val] s) hlt
  rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hab) with h | h
  · exact ⟨a, b, h, Nat.lt_succ_iff.mp b.isLt, he⟩
  · exact ⟨b, a, h, Nat.lt_succ_iff.mp a.isLt, he.symm⟩

end V4N
