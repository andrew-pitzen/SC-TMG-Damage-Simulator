"""Helpers that decide which unit toggles, passives and upgrades apply. No Streamlit imports,
so they can be unit-tested.

units.json conventions
  unit["toggles"]  : [{"name": "Siege Mode", "label": "...", "default": false, "size_when_on": 3}]
  weapon["requires"]: {"Siege Mode": true}   -> weapon only offered when the toggle matches
  unit["options"]  : passives / upgrades / abilities that change attacks or defence:
      name, kind ("passive" | "upgrade" | "ability"), side ("attack" | "defend"),
      effects {...}                      (keys of AttackMods / DefendMods)
      keywords  ["Burst Fire 4\" (3)", "Concentrated Fire (1)"]   attack options only: keywords the option
                                         adds to the weapon. They are merged with the weapon's own
                                         keywords (weapon_keywords); Burst Fire, Long Range and
                                         Concentrated Fire are acted on, others are display-only.
      phase     "any" | "ranged" | "melee"   attack phase of the weapon the option works with
      weapons   [names]                  only for weapons whose name contains one of these
      when      {"target_size_min": 3, "target_tag": "Armoured", "toggle": {"Siege Mode": false}}
      situational true                   needs a board condition the app cannot see (off by default)
      auto      true                     no checkbox: applied automatically whenever 'when' is met
      note      free text
      scope     "all"  -> applies to EVERY weapon (main weapon, Sidearms) and to Impact rolls
                "first" (default for defender options and for cards) -> main weapon only
                attack options without a scope stay per-weapon (ticked under the weapon they work with);
                with "all" they are shown once and applied to every weapon in the attack
      impact_dice  N   attack option: +N Impact dice per model (shown in the Impact section only)
  unit["auto_hits"]: [{"label": "Shadow Strike", "phase": "melee", "dice": 4, "dmg": 1, "verify": "..."}]
                     abilities whose dice hit automatically (no Hit roll); each is its own extra attack with
                     its own defender box. A single object instead of a list also works.
  unit["direct_damage"]: [{"label": "Ability name", "phase": "any", "dmg": 2, "note": "..."}]
                     abilities that deal damage directly (no dice: no Hit, Armour or Evade, and not reduced
                     by Total damage reduction). One object instead of a list also works.
  unit["impact"]   : {"dice": 1, "hit": 4, "dmg": 1, "per_model": true, "verify": "..."}  Impact rolls for
                     a close combat attack: a separate extra attack, no attacker keywords/cards, only
                     defender effects with scope "all".
  cards.json cards may set "affects": "opponent" -> a debuff on the enemy: an attacker card then holds
                     DEFENDER effects (e.g. armour_mod -1) and a defender card holds ATTACKER effects
                     (e.g. hit_mod -1); scope and "applies to" work as usual.
  cards.json attacker cards may set "targets": ["impact"] -> the card affects ONLY Impact rolls (no weapon
                     picker; ignored while Impact is off).
  effects may contain "dmg_set_target_size": true  -> damage equals the target's Size (min 1)
"""
from __future__ import annotations

import re

from engine import DefendMods, combine, norm_tag

KIND_LABEL = {"passive": "Passive", "upgrade": "Upgrade", "ability": "Ability"}

# Defender numbers that belong to the attack as a whole (casualties), never to one weapon.
ATTACK_WIDE = {"total_reduction", "hp_bonus", "shield_bonus"}


# --------------------------------------------------------------------------- #
# Scope: does a card / option affect only the main weapon, or every weapon?
# --------------------------------------------------------------------------- #
def scope_of(item: dict, default: str = "first") -> str:
    return item.get("scope", default)


def is_all_scope(item: dict) -> bool:
    return item.get("scope") == "all"


def auto_hit_list(unit: dict) -> list:
    """The unit's auto-hit abilities as a list. units.json may hold one object or a list of them."""
    ah = unit.get("auto_hits")
    if not ah:
        return []
    return list(ah) if isinstance(ah, list) else [ah]


def direct_damage_list(unit: dict) -> list:
    """The unit's automatic damage abilities as a list (one object or a list in units.json)."""
    dd = unit.get("direct_damage")
    if not dd:
        return []
    return list(dd) if isinstance(dd, list) else [dd]


def is_impact_option(opt: dict) -> bool:
    return "impact_dice" in opt


def is_opponent_card(card: dict) -> bool:
    """A card played by one side whose effects land on the OTHER side (a debuff on the enemy):
    an attacker card with defender effects (e.g. Armour -1) or a defender card with attacker effects
    (e.g. Hit -1 on the attacking unit)."""
    return card.get("affects") == "opponent"


def card_targets(card: dict) -> list:
    """Attack ids a card is fixed to (e.g. ["impact"]); [] for ordinary cards."""
    return list(card.get("targets", []))


def card_label(name: str, card: dict) -> str:
    if "impact" in card_targets(card):
        return f"{name}  [Impact only]"
    return f"{name}  [all weapons]" if is_all_scope(card) else name


def defend_mods_by_scope(items) -> tuple:
    """items: [(scope, effects dict)] with scope "all" or "first".

    Returns (shared, main, other):
      shared - attack-wide mods used for casualties (all-scope effects + total reduction / HP / Shield
               bonus from every effect)
      main   - what the main weapon's rolls see (all-scope + first-scope effects)
      other  - what Sidearms and Impact rolls see (all-scope effects only)
    """
    all_fx = [e for s, e in items if s == "all"]
    first_fx = [e for s, e in items if s != "all"]
    wide = [{k: v for k, v in e.items() if k in ATTACK_WIDE} for e in first_fx]
    return (combine(all_fx + wide, DefendMods),
            combine(all_fx + first_fx, DefendMods),
            combine(all_fx, DefendMods))


def defend_mods_by_target(items, target_ids) -> tuple:
    """Defender effects worked out per attack in the pool (main weapon, each Sidearm, Impact).

    items: [(scope, effects dict, assigned target ids)]. A scope "all" item reaches every target;
    any other item reaches only the targets listed in `assigned`.

    Returns (shared, {target_id: DefendMods}):
      shared - attack-wide mods used for casualties (all-scope effects + total reduction / HP / Shield
               bonus from every effect, wherever the card was assigned)
      per    - what that target's own rolls see (all-scope + the effects assigned to it)
    """
    all_fx = [e for s, e, _ in items if s == "all"]
    scoped = [(e, a) for s, e, a in items if s != "all"]
    wide = [{k: v for k, v in e.items() if k in ATTACK_WIDE} for e, _ in scoped]
    shared = combine(all_fx + wide, DefendMods)
    per = {t: combine(all_fx + [e for e, a in scoped if t in a], DefendMods) for t in target_ids}
    return shared, per


# --------------------------------------------------------------------------- #
# Keywords (weapon keywords + keywords granted by ticked options)
# --------------------------------------------------------------------------- #
def weapon_keywords(weapon: dict | None, options=()) -> list:
    """The weapon's own keywords plus any keywords granted by the given (ticked) options."""
    kws = list((weapon or {}).get("keywords", []))
    for o in options:
        kws += o.get("keywords", [])
    return kws


def find_keyword(kws: list, name: str) -> str | None:
    """First keyword containing `name` (case-insensitive), or None."""
    name = name.lower()
    return next((k for k in kws if name in k.lower()), None)


def keyword_value(kws: list, name: str, pick=max) -> int:
    """Number in a keyword's trailing parentheses, e.g. 'Concentrated Fire (1)' -> 1 and
    'Burst Fire 4" (3)' -> 3. 0 when the keyword is missing or has no number. If several
    keywords match, `pick` (max or min) chooses between their values."""
    vals = []
    for k in kws:
        if name.lower() in k.lower():
            m = re.search(r"\((\d+)\)\s*$", k)
            if m:
                vals.append(int(m.group(1)))
    return pick(vals) if vals else 0


# --------------------------------------------------------------------------- #
# Swapping attacker and defender
# --------------------------------------------------------------------------- #
def swap_selection(state, units: dict, custom: str) -> list:
    """Swap the faction and unit picks of the attacker and the defender.

    `state` is st.session_state (any dict-like). Keys: a_fac / d_fac and a_unit_<faction> /
    d_unit_<faction>. Units without weapons (e.g. a Pylon) can't be attackers; in that case the
    attacker keeps the faction but falls back to its first armed unit. Returns notes to show.
    """
    a_fac, d_fac = state.get("a_fac"), state.get("d_fac")
    a_unit = state.get(f"a_unit_{a_fac}")
    d_unit = state.get(f"d_unit_{d_fac}")
    notes = []
    state["a_fac"], state["d_fac"] = d_fac, a_fac

    # the old defender becomes the attacker
    if d_fac != custom:
        if d_unit in units[d_fac] and units[d_fac][d_unit]["weapons"]:
            state[f"a_unit_{d_fac}"] = d_unit
        else:
            state.pop(f"a_unit_{d_fac}", None)
            if d_unit:
                notes.append(f"{d_unit} has no weapons, so it can't be the attacker; showing the first armed {d_fac} unit.")
    # the old attacker becomes the defender
    if a_fac != custom:
        if a_unit in units[a_fac]:
            state[f"d_unit_{a_fac}"] = a_unit
        else:
            state.pop(f"d_unit_{a_fac}", None)
    return notes


def toggles_match(requires: dict | None, state: dict) -> bool:
    return all(bool(state.get(k, False)) == bool(v) for k, v in (requires or {}).items())


def unit_size(unit: dict, state: dict) -> int:
    size = unit.get("size", 2)
    for t in unit.get("toggles", []):
        if state.get(t["name"], False) and t.get("size_when_on") is not None:
            size = t["size_when_on"]
    return size


def phase_ok(opt: dict, attack_phase: str) -> bool:
    return opt.get("phase", "any") in ("any", attack_phase)


def matches_weapon(opt: dict, weapon: dict | None) -> bool:
    if is_impact_option(opt):
        return False          # Impact upgrades are shown in the Impact section, not under a weapon
    if weapon is None:
        return True
    if not phase_ok(opt, weapon["phase"]):
        return False
    names = opt.get("weapons")
    return not names or any(n in weapon["name"] for n in names)


def when_met(opt: dict, own_state: dict, target_size: int, target_tags) -> bool:
    cond = opt.get("when") or {}
    if not toggles_match(cond.get("toggle"), own_state):
        return False
    if "target_size_min" in cond and target_size < cond["target_size_min"]:
        return False
    if "target_tag" in cond and norm_tag(cond["target_tag"]) not in {norm_tag(t) for t in target_tags}:
        return False
    return True


def default_checked(opt: dict) -> bool:
    return opt.get("kind") == "passive" and not opt.get("situational")


def option_label(opt: dict) -> str:
    tag = KIND_LABEL.get(opt.get("kind", ""), "")
    return f"{opt['name']} [{tag}]" if tag else opt["name"]


def option_effects(opt: dict, target_size: int) -> dict:
    eff = dict(opt["effects"])
    if eff.pop("dmg_set_target_size", False):
        eff["dmg_set"] = max(1, int(target_size))
    return eff
