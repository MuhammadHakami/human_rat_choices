/-
  Vision4Neuro stage 5 (2026-11-06): why the mushroom-body circuit transfers less to never-seen rooms, and what a
  room-invariant place drive changes. Readout W (m outputs x K Kenyon cells), codes z_A, z_B, three-factor update
  ΔW = η δ g z_Aᵀ (g = post-synaptic factor, δ = dopamine-like TD error).
  1. transfer_identity: after the update at state A, the output at ANY state B moves by η δ g ⟨z_A, z_B⟩: transfer to a
     never-seen room is exactly proportional to the code overlap.
  2. self_interference: at A itself the move is η δ g ‖z_A‖²: states with identical views (old and new goal in this
     task) receive the full update whatever the sparsity, so sparseness cannot protect them; only a context gate can.
  3. drive_inner / drive_diff: adding a common drive β c to both pre-activations leaves their difference unchanged and
     adds β (⟨a,c⟩ + ⟨c,b⟩) + β² ‖c‖² to their inner product: a room-invariant place input raises cross-room overlap.
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

end V4N
