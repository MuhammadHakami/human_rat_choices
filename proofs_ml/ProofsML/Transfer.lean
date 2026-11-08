/-
  Vision4Neuro stages 5-6 (2026-11-06/07): why the mushroom-body circuit transfers less to never-seen rooms, and what a
  room-invariant place drive changes. Readout W (m outputs x K Kenyon cells), codes z_A, z_B, three-factor update
  ΔW = η δ g z_Aᵀ (g = post-synaptic factor, δ = dopamine-like TD error).
  1. transfer_identity: after the update at state A, the output at ANY state B moves by η δ g ⟨z_A, z_B⟩: transfer to a
     never-seen room is exactly proportional to the code overlap.
  2. self_interference: at A itself the move is η δ g ‖z_A‖²: states with identical views (old and new goal in this
     task) receive the full update whatever the sparsity, so sparseness cannot protect them; only a context gate can.
  3. drive_inner / drive_diff: adding a common drive β c to both pre-activations leaves their difference unchanged and
     adds β (⟨a,c⟩ + ⟨c,b⟩) + β² ‖c‖² to their inner product: a room-invariant place input raises cross-room overlap.
  4. concat_inner: a readout on the joined code [z, g u] transfers by <z_A,z_B> + g^2 <u_A,u_B>; copies_orthogonal:
     context copies of a place code are orthogonal across contexts (stage 6).
-/
import Mathlib

open Finset

namespace V4N

/-- Three-factor update of the readout at state A. -/
def update {m K : ℕ} (W : Fin m → Fin K → ℝ) (η δ : ℝ) (g : Fin m → ℝ) (zA : Fin K → ℝ) : Fin m → Fin K → ℝ :=
  fun i j => W i j + η * δ * g i * zA j

/-- Output i of a readout at code z. -/
def out {m K : ℕ} (W : Fin m → Fin K → ℝ) (z : Fin K → ℝ) (i : Fin m) : ℝ := ∑ j, W i j * z j

theorem transfer_identity {m K : ℕ} (W : Fin m → Fin K → ℝ) (η δ : ℝ) (g : Fin m → ℝ) (zA zB : Fin K → ℝ) (i : Fin m) :
    out (update W η δ g zA) zB i = out W zB i + η * δ * g i * ∑ j, zA j * zB j := by
  simp only [out, update, add_mul, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1; apply Finset.sum_congr rfl; intro j _; ring

theorem self_interference {m K : ℕ} (W : Fin m → Fin K → ℝ) (η δ : ℝ) (g : Fin m → ℝ) (zA : Fin K → ℝ) (i : Fin m) :
    out (update W η δ g zA) zA i = out W zA i + η * δ * g i * ∑ j, zA j ^ 2 := by
  rw [transfer_identity]; congr 2; apply Finset.sum_congr rfl; intro j _; ring

theorem drive_diff {K : ℕ} (a b c : Fin K → ℝ) (β : ℝ) (j : Fin K) : (a j + β * c j) - (b j + β * c j) = a j - b j := by ring

theorem drive_inner {K : ℕ} (a b c : Fin K → ℝ) (β : ℝ) :
    ∑ j, (a j + β * c j) * (b j + β * c j) =
      ∑ j, a j * b j + β * (∑ j, a j * c j + ∑ j, c j * b j) + β ^ 2 * ∑ j, c j ^ 2 := by
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]; apply Finset.sum_congr rfl; intro j _; ring

/-- Stage 6: a readout on the joined code [z, g u] transfers by the Kenyon overlap plus g² times the place overlap. -/
theorem concat_inner {K N : ℕ} (zA zB : Fin K → ℝ) (uA uB : Fin N → ℝ) (g : ℝ) :
    ∑ j, Fin.append zA (fun k => g * uA k) j * Fin.append zB (fun k => g * uB k) j =
      ∑ j, zA j * zB j + g ^ 2 * ∑ k, uA k * uB k := by
  rw [Fin.sum_univ_add]; simp only [Fin.append_left, Fin.append_right, Finset.mul_sum]
  congr 1; apply Finset.sum_congr rfl; intro k _; ring

/-- Context copy of a place code: b on the copy of context c, zero on the other copies. -/
def ctxCopy {C N : ℕ} (c : Fin C) (b : Fin N → ℝ) : Fin C → Fin N → ℝ := fun c' k => if c' = c then b k else 0

/-- Copies for different contexts are orthogonal: learning in one context cannot move the readout in the other. -/
theorem copies_orthogonal {C N : ℕ} (c d : Fin C) (h : c ≠ d) (b b' : Fin N → ℝ) :
    ∑ c', ∑ k, ctxCopy c b c' k * ctxCopy d b' c' k = 0 := by
  apply Finset.sum_eq_zero; intro c' _; apply Finset.sum_eq_zero; intro k _
  unfold ctxCopy; by_cases h1 : c' = c
  · subst h1; simp [h]
  · simp [h1]

end V4N
