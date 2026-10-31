"""Monte Carlo engine for a StarCraft: Tabletop Miniatures Game attack (ranged or close combat).

Follows the Quick Start Rules attack resolution (8.7.4):
  Attack Pool -> Roll to Hit -> Surge -> Armour Rolls -> Evade Rolls -> Damage & Casualties

Rules implemented:
  * Target numbers are modified (never the die), clamped to 2+..6+.
  * Natural 6 always succeeds, natural 1 always fails. A die may be re-rolled once.
  * PRECISION (X): after the hit roll, up to X failed dice become hits.
  * SURGE: S die (D3 = roll D6, halve, round up; "D3+1" adds 1 to the result) moves that many
    dice from the Armour Pool to the Damage Pool (or the Evade Pool if the defender may evade)
    if the target has a Combat Tag matching the weapon's Surge Type. It can never move more
    dice than the Armour Pool holds.
  * BLAST TEMPLATE: the caller sets RoA = base + BT; an S Dice of "BT" moves exactly BT dice.
  * CRITICAL HIT (X): moves up to X more dice from the Armour Pool straight to the Damage Pool.
  * DODGE (X): reduces the dice moved by Surge/Critical Hit by X (min 0); those dice are then
    armour-rolled normally.
  * Armour: a die >= Armour value is saved. TOUGH (X): up to X failed saves become successes.
  * Evade (only if eligible): dice >= Evade value are removed. ANTI-EVADE (X) worsens it by X.
  * Damage = remaining dice x DMG (PIERCE tag (X) replaces DMG vs. that tag).
  * Casualties: total damage (plus damage already on the unit) removes models in order; the
    first model carries any Shield; leftover damage stays as a marker.

Keyword definitions for PRECISION, CRITICAL HIT, TOUGH, DODGE and ANTI-EVADE follow the
community rules wiki (tmgcraft.com); step ordering between Surge and Critical Hit is inferred.
Numeric keywords never stack: the highest value is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, List

import numpy as np

N_DEFAULT = 20_000

# Keywords never stack: use the highest value from weapon / cards.
KEYWORD_MAX = {"precision", "critical", "anti_evade", "pierce_x", "tough", "dodge", "dmg_set"}


def norm_tag(t: str) -> str:
    t = t.strip().lower()
    return "armoured" if t == "armored" else t


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Weapon:
    models: int = 3            # models firing this weapon profile
    roa: int = 2               # Rate of Attack (dice per model)
    hit: int = 4               # Hit target number (4 = "4+")
    dmg: int = 1
    surge_die: str = "None"    # "None" | "D3" | "D6"
    surge_plus: int = 0        # "D3+1" -> surge_die "D3", surge_plus 1
    surge_tags: List[str] = field(default_factory=list)  # triggers on ANY listed tag
    pierce_x: int = 0          # PIERCE (X): damage X instead of DMG ...
    pierce_tag: str = ""       # ... against units with this Combat Tag
    precision: int = 0
    critical: int = 0
    anti_evade: int = 0
    bt: int = 0                # Blast Template: units under the template (S Dice "BT" = this value)


@dataclass
class Defender:
    tags: List[str] = field(default_factory=list)
    armour: int | None = 5     # target number; None = null ("-"), always fails
    evade: int | None = None   # target number; None = null
    evade_eligible: bool = False  # Engaged target / high ground / ability grants a roll
    hp: int = 3                # hit points per model
    shield: int = 0            # added to the first model's HP
    models: int = 5
    visible: int | None = None # visible models (None = all). Caps casualties.
    existing_damage: int = 0


@dataclass
class AttackMods:
    hit_mod: int = 0            # +1 = easier (4+ -> 3+)
    bonus_dice_per_model: int = 0
    bonus_dice_flat: int = 0
    dmg_mod: int = 0
    dmg_set: int = 0            # treat the weapon's DMG as this value (0 = off)
    surge_bonus: int = 0        # added to the surge die result
    set_surge_die: str = ""     # replace the weapon's S Dice ("D6")
    grant_surge_die: str = ""   # give a weapon with no S Dice a die ("D3"/"D6")
    grant_surge_plus: int = 0
    grant_surge_tags: List[str] = field(default_factory=list)
    reroll_hits: str = "none"   # "none" | "ones" | "misses"
    precision: int = 0
    critical: int = 0
    anti_evade: int = 0
    pierce_x: int = 0
    pierce_tag: str = ""


@dataclass
class DefendMods:
    armour_mod: int = 0         # +1 = easier save
    evade_mod: int = 0
    grant_evade: int = 0        # grants an Evade roll at this value (0 = none)
    reroll_armour: str = "none" # "none" | "ones" | "fails"
    reroll_evade: str = "none"
    tough: int = 0
    dodge: int = 0
    damage_reduction: int = 0   # per unsaved die (min 0 per die)
    total_reduction: int = 0    # off the total damage of the attack (min 0)
    attack_pool_reduction: int = 0  # fewer attack dice (e.g. Guardian Shield)
    shield_bonus: int = 0
    hp_bonus: int = 0           # per model
    ignore_surge: bool = False
    grant_evade_roll: bool = False  # eligible to Evade using the unit's own Evade value
    armour_set: int = 0         # Armour is treated as this target number (0 = off)


def combine(effects: List[dict], cls):
    """Merge card effect dicts into one mods dataclass.

    Numbers add up, except keywords (precision, critical, tough, dodge, anti_evade,
    pierce_x) which take the highest; reroll modes take the strongest.
    """
    mods = cls()
    order = {"none": 0, "ones": 1, "misses": 2, "fails": 2}
    names = {f.name for f in fields(cls)}
    for eff in effects:
        for key, val in eff.items():
            if key not in names:
                continue
            cur = getattr(mods, key)
            if key.startswith("reroll"):
                if order.get(val, 0) > order.get(cur, 0):
                    setattr(mods, key, val)
            elif key in KEYWORD_MAX:
                if val > cur:
                    setattr(mods, key, val)
            elif key in ("pierce_tag", "grant_surge_die", "set_surge_die"):
                if val:
                    setattr(mods, key, val)
            elif key == "grant_surge_tags":
                setattr(mods, key, sorted(set(cur) | set(val)))
            elif key in ("ignore_surge", "grant_evade_roll"):
                setattr(mods, key, bool(cur or val))
            elif key == "armour_set":
                if val and (cur == 0 or val < cur):  # best (lowest) target number
                    setattr(mods, key, val)
            elif key == "grant_evade":
                if val and (cur == 0 or val < cur):  # best (lowest) target number
                    setattr(mods, key, val)
            else:
                setattr(mods, key, cur + val)
    return mods


# --------------------------------------------------------------------------- #
# Dice helpers
# --------------------------------------------------------------------------- #
def clamp_target(t: int) -> int:
    return max(2, min(6, t))


def _success(rolls: np.ndarray, target: int | None) -> np.ndarray:
    """Test against a target number. Natural 6 succeeds, natural 1 fails."""
    if target is None:
        return np.zeros(rolls.shape, dtype=bool)
    return ((rolls >= target) | (rolls == 6)) & (rolls != 1)


def roll_pool(counts: np.ndarray, target: int | None, reroll: str,
              rng: np.random.Generator) -> np.ndarray:
    """Roll counts[i] D6 for each trial i. Returns successes per trial.

    reroll: "none", "ones" (reroll natural 1s) or "misses"/"fails" (reroll all failures).
    A die may only be re-rolled once, and the new result stands.
    """
    n = len(counts)
    width = int(counts.max()) if n else 0
    if width == 0:
        return np.zeros(n, dtype=np.int64)
    live = np.arange(width)[None, :] < counts[:, None]
    rolls = rng.integers(1, 7, size=(n, width))
    ok = _success(rolls, target)
    if reroll != "none":
        redo = (~ok) if reroll in ("misses", "fails") else (rolls == 1)
        redo &= live
        new = rng.integers(1, 7, size=(n, width))
        ok = np.where(redo, _success(new, target), ok)
    return (ok & live).sum(axis=1).astype(np.int64)


def roll_surge(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "D6":
        return rng.integers(1, 7, size=n)
    if kind == "D3":  # roll a D6, halve, round up
        return (rng.integers(1, 7, size=n) + 1) // 2
    return np.zeros(n, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(weapon: Weapon, defender: Defender,
             atk: AttackMods | None = None, dfn: DefendMods | None = None,
             n: int = N_DEFAULT, seed: int | None = None) -> Dict[str, np.ndarray]:
    atk = atk or AttackMods()
    dfn = dfn or DefendMods()
    rng = np.random.default_rng(seed)
    tags = {norm_tag(t) for t in defender.tags}

    # 1. Attack pool
    pool = weapon.models * max(0, weapon.roa + atk.bonus_dice_per_model) + atk.bonus_dice_flat
    pool = max(0, pool - dfn.attack_pool_reduction)
    attack_pool = np.full(n, pool, dtype=np.int64)

    # 2. Roll to hit, then PRECISION promotes failed dice to hits
    hit_target = clamp_target(weapon.hit - atk.hit_mod)
    hits = roll_pool(attack_pool, hit_target, atk.reroll_hits, rng)
    precision = max(weapon.precision, atk.precision)
    if precision:
        hits = hits + np.minimum(precision, attack_pool - hits)

    # 3. Surge + Critical Hit (both bypass armour; Dodge reduces what they move)
    surge_die, surge_plus, surge_tags = weapon.surge_die, weapon.surge_plus, list(weapon.surge_tags)
    if atk.set_surge_die and surge_die != "None":
        surge_die = atk.set_surge_die
    if surge_die == "None" and atk.grant_surge_die:
        surge_die, surge_plus = atk.grant_surge_die, atk.grant_surge_plus
        surge_tags = list(atk.grant_surge_tags) or surge_tags
    surge_tags_n = {norm_tag(t) for t in surge_tags}
    surge_applies = (surge_die != "None" and not dfn.ignore_surge and bool(surge_tags_n & tags))
    if surge_applies:
        base_surge = (np.full(n, weapon.bt, dtype=np.int64) if surge_die == "BT"
                      else roll_surge(surge_die, n, rng))
        surge_roll = np.maximum(0, base_surge + surge_plus + atk.surge_bonus)
    else:
        surge_roll = np.zeros(n, dtype=np.int64)
    critical = max(weapon.critical, atk.critical)
    bypass = np.minimum(surge_roll + critical, hits)
    bypass = np.maximum(0, bypass - dfn.dodge) if (surge_applies or critical) else bypass
    armour_pool = hits - bypass

    # 4. Armour rolls; TOUGH turns up to X failures into saves
    base_armour = defender.armour
    if base_armour is not None and dfn.armour_set:
        base_armour = dfn.armour_set
    armour_target = None if base_armour is None else clamp_target(base_armour - dfn.armour_mod)
    saved = roll_pool(armour_pool, armour_target, dfn.reroll_armour, rng)
    armour_fail = armour_pool - saved
    if dfn.tough and armour_target is not None:
        tough_saves = np.minimum(dfn.tough, armour_fail)
        saved = saved + tough_saves
        armour_fail = armour_fail - tough_saves

    # 5. Evade rolls (bypassed dice skip armour but may still be evaded)
    evade_value = defender.evade
    if dfn.grant_evade:
        evade_value = dfn.grant_evade if evade_value is None else min(evade_value, dfn.grant_evade)
    evade_active = ((defender.evade_eligible or bool(dfn.grant_evade) or dfn.grant_evade_roll)
                    and evade_value is not None)
    pre_evade = armour_fail + bypass
    ev_target = None
    if evade_active:
        ev_target = clamp_target(evade_value - dfn.evade_mod + max(weapon.anti_evade, atk.anti_evade))
        evaded = roll_pool(pre_evade, ev_target, dfn.reroll_evade, rng)
    else:
        evaded = np.zeros(n, dtype=np.int64)
    damage_dice = pre_evade - evaded

    # 6. Damage & casualties
    per_die = atk.dmg_set if atk.dmg_set else weapon.dmg
    pierce_x = max(weapon.pierce_x, atk.pierce_x)
    pierce_tag = norm_tag(weapon.pierce_tag or atk.pierce_tag)
    if pierce_x and pierce_tag in tags:
        per_die = pierce_x
    per_die = max(0, per_die + atk.dmg_mod - dfn.damage_reduction)
    new_damage = np.maximum(0, damage_dice * per_die - dfn.total_reduction)
    total = new_damage + defender.existing_damage

    hp = max(1, defender.hp + dfn.hp_bonus)
    first_hp = max(1, hp + defender.shield + dfn.shield_bonus)
    visible = defender.models if defender.visible is None else min(defender.visible, defender.models)
    kills = np.where(total >= first_hp, 1 + (total - first_hp) // hp, 0)
    kills = np.minimum(kills, visible)
    spent = np.where(kills > 0, first_hp + (kills - 1) * hp, 0)
    leftover = np.where(kills >= visible, 0, total - spent)

    return {
        "attack_pool": attack_pool, "hits": hits, "surged": bypass,
        "armour_pool": armour_pool, "armour_saved": saved, "armour_fail": armour_fail,
        "evaded": evaded, "damage_dice": damage_dice, "damage": new_damage,
        "kills": kills, "leftover": leftover,
        "meta": {
            "pool": pool, "hit_target": hit_target, "armour_target": armour_target,
            "evade_target": ev_target, "surge_die": ((f"BT={weapon.bt}" if surge_die == "BT" else surge_die) + (f"+{surge_plus + atk.surge_bonus}" if surge_plus + atk.surge_bonus else "")) if surge_applies else "off",
            "per_die_damage": per_die, "first_model_hp": first_hp,
            "precision": precision, "critical": critical, "tough": dfn.tough, "dodge": dfn.dodge,
        },
    }
