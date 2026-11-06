"""ArcFly: a fast, LEARNABLE, biologically-motivated ARC-AGI-3 fly task.

Re-embodies ARC-AGI-3's three named puzzle families in Drosophila sensorimotor biology.
All three share the SAME 11-D observation and action space, so one learner architecture
solves the whole suite (each puzzle adds a mechanic, mirroring ARC-AGI-3 levels):

  * "locksmith"    (ls20, "Agent reasoning") — reach the odor-tagged BUTTON (press =
                   rotate the key symbol to match the goal / unlock the food box), then
                   reach the distinct-odor FOOD and eat it.
  * "sequence"     (ft09, "Elementary Logic") — TWO buttons; BOTH must be pressed before
                   the food box opens (order-free logical AND).
  * "orchestration"(vc33, "Orchestration") — button -> food, but an AVERSIVE odor source
                   (bitter/CO2) must be AVOIDED; contacting it costs a life.

Sensing: the fly compares odor bilaterally (left/right antenna) among random distractor
odors. Movement is bounded by a MOVE budget and 3 LIVES (over-moving costs a life) — the
ARC three-life / relative-action-efficiency mechanic. The policy is LEARNED, not scripted.

Fast 2-D point-fly for training/comparison; the learned policy is later embodied in flygym.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

PUZZLES = ("locksmith", "sequence", "orchestration")


class ArcFlyEnv(gym.Env):
    def __init__(self, budget=80, lives=3, arena=6.0, sigma=8.0, n_distractor=2, seed=0,
                 food_reward=25.0, button_reward=5.0, puzzle="locksmith",
                 retry_on_life_loss=False, n_symbols=4, shaped=True, food_gated=False):
        super().__init__()
        assert puzzle in PUZZLES, puzzle
        self.puzzle = puzzle
        self.budget, self.lives0, self.arena, self.sigma, self.nd = budget, lives, arena, sigma, n_distractor
        # food_reward is the appetitive "sugar" reinforcement; ground it in the Eon
        # connectome's sugar-GRN cascade via arc_fly.eon_bridge (default 25.0 preserves
        # the released checkpoints). button_reward marks the symbol-match sub-goal.
        self.food_reward, self.button_reward = food_reward, button_reward
        # retry_on_life_loss=True is the faithful ARC-AGI-3 mechanic: on GAME_OVER the level
        # RESETS (fly back to start, key symbol un-rotated), keeping the SAME layout, so a
        # learner can adapt across the 3 lives. Default False preserves the released checkpoints.
        self.retry_on_life_loss = retry_on_life_loss
        self.shaped = shaped                       # False = sparse reward (button/food only), no distance shaping
        # food_gated=True: the food box is SEALED (emits ~10% odor) until the button opens it,
        # so the innate odor landscape itself guides the button->food sequence. This is the
        # biologically-natural "Drosophila puzzle environment" that maximizes ZERO-SHOT success.
        self.food_gated = food_gated
        self.n_symbols = n_symbols                 # the key symbol cycles through this many shapes
        self._rng = np.random.default_rng(seed)
        # symbols drawn from a SEPARATE stream so the layout RNG is byte-identical to the
        # released checkpoints (adding symbol draws to _rng would shift every later layout).
        self._sym_rng = np.random.default_rng(seed + 917)
        self.step_len = 0.5
        # obs: [Lb,Rb, Lf,Rf, L3,R3, cos h, sin h, unlock_frac, moves/budget, lives/lives0]
        #   channel 3 = button2 (sequence) / aversive (orchestration) / distractor (locksmith)
        self.observation_space = spaces.Box(-1, 1, (11,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)   # turn

    # ---- odor field ----
    def _conc(self, src, p):
        return float(np.exp(-np.sum((p - src) ** 2) / (2 * self.sigma ** 2)))

    def _ch3_sources(self):
        if self.puzzle == "sequence":
            return [self.button2]
        if self.puzzle == "orchestration":
            return [self.aversive]
        return self.distractors

    def _sense(self):
        perp = np.array([-np.sin(self.h), np.cos(self.h)])
        la, ra = self.pos + 0.4 * perp, self.pos - 0.4 * perp
        # sealed food box leaks little odor until the button opens it (food_gated)
        fg = 0.1 if (self.food_gated and not self._unlocked) else 1.0
        out = [self._conc(self.button, la), self._conc(self.button, ra),
               fg * self._conc(self.food, la), fg * self._conc(self.food, ra)]
        c3 = self._ch3_sources()
        out += [max(self._conc(s, la) for s in c3), max(self._conc(s, ra) for s in c3)]
        return np.array(out, np.float32)

    def _unlock_frac(self):
        if self.puzzle == "sequence":
            return 0.5 * (int(self.pressed) + int(self.pressed2))
        return 1.0 if self.pressed else 0.0

    def _obs(self):
        s = self._sense()
        return np.concatenate([s, [np.cos(self.h), np.sin(self.h), self._unlock_frac(),
                                   self.moves / self.budget, self.lives / self.lives0]]).astype(np.float32)

    def _rand_pt(self):
        return self._rng.uniform(-self.arena, self.arena, 2)

    @property
    def _unlocked(self):
        return (self.pressed and self.pressed2) if self.puzzle == "sequence" else self.pressed

    def _active_target(self):
        """The waypoint the fly should currently pursue (for potential-based shaping)."""
        if self.puzzle == "sequence" and not self._unlocked:
            cands = ([] if self.pressed else [self.button]) + ([] if self.pressed2 else [self.button2])
            return min(cands, key=lambda s: np.linalg.norm(s - self.pos))
        if not self._unlocked:
            return self.button
        return self.food

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self._rand_pt(); self.h = self._rng.uniform(-np.pi, np.pi)
        self.button = self._rand_pt(); self.food = self._rand_pt()
        self.distractors = [self._rand_pt() for _ in range(self.nd)]
        self.button2 = self._rand_pt() if self.puzzle == "sequence" else None
        self.aversive = self._rand_pt() if self.puzzle == "orchestration" else None
        # ARC-AGI-3 symbol-rotation mechanic: the key symbol must be rotated to the goal
        # symbol; pressing the button rotates it to match (see step()).
        self.goal_symbol = int(self._sym_rng.integers(self.n_symbols))
        self.key_symbol0 = (self.goal_symbol + 1 + int(self._sym_rng.integers(self.n_symbols - 1))) % self.n_symbols
        self.key_symbol = self.key_symbol0
        self.start_pos = self.pos.copy(); self.start_h = self.h   # for retry-to-start
        self.life_index = 0
        self.pressed = False; self.pressed2 = False
        self.moves = 0; self.lives = self.lives0
        self._pdist = np.linalg.norm(self._active_target() - self.pos)
        return self._obs(), {}

    def _retry(self):
        """ARC-AGI-3 RESET on GAME_OVER: fly back to start, symbol un-rotated, SAME layout."""
        self.pos = self.start_pos.copy(); self.h = self.start_h
        self.pressed = False; self.pressed2 = False
        self.key_symbol = self.key_symbol0
        self.moves = 0; self.life_index += 1
        self._pdist = np.linalg.norm(self._active_target() - self.pos)

    def _lose_life(self, reward):
        """Decrement a life; retry the level or terminate. Returns (obs, reward, terminated, info)."""
        self.lives -= 1
        info = {"success": False, "life_lost": True, "life_index": self.life_index}
        if self.lives <= 0:
            return self._obs(), reward, True, info
        if self.retry_on_life_loss:
            self._retry()
        else:
            self.moves = 0
        return self._obs(), reward, False, info

    def step(self, action):
        turn = float(np.clip(action[0], -1, 1)) * 0.6
        self.h = (self.h + turn + np.pi) % (2 * np.pi) - np.pi
        self.pos = np.clip(self.pos + self.step_len * np.array([np.cos(self.h), np.sin(self.h)]),
                           -self.arena, self.arena)
        self.moves += 1
        reward = -0.01
        # potential-based shaping toward the current waypoint (disabled in sparse mode, so
        # zero-shot fails on life 1 and within-episode learning must discover the mechanic)
        dist = np.linalg.norm(self._active_target() - self.pos)
        if self.shaped:
            reward += 1.2 * (self._pdist - dist)
        self._pdist = dist

        # aversive contact (orchestration) — costs a life
        if self.puzzle == "orchestration" and np.linalg.norm(self.aversive - self.pos) < 0.9:
            reward -= 5.0
            o, r, term, info = self._lose_life(reward)
            return o, r, term, False, info

        # press button(s) -> rotate the key symbol to match the goal symbol
        if not self._unlocked:
            if not self.pressed and np.linalg.norm(self.button - self.pos) < 0.9:
                self.pressed = True; self.key_symbol = self.goal_symbol   # symbol rotated to match
                reward += self.button_reward
                self._pdist = np.linalg.norm(self._active_target() - self.pos)
            if self.puzzle == "sequence" and not self.pressed2 and np.linalg.norm(self.button2 - self.pos) < 0.9:
                self.pressed2 = True; reward += self.button_reward
                self._pdist = np.linalg.norm(self._active_target() - self.pos)
        elif np.linalg.norm(self.food - self.pos) < 0.9:
            reward += self.food_reward                          # ate the food (Eon sugar-GRN reinforcement)
            return self._obs(), reward, True, False, {"success": True, "life_index": self.life_index}

        if self.moves >= self.budget:                           # spent the move budget -> lose a life
            reward -= 3.0
            o, r, term, info = self._lose_life(reward)
            return o, r, term, False, info
        return self._obs(), reward, False, False, {"success": False}
