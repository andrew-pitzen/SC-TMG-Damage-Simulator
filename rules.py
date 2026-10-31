"""Helpers that decide which unit toggles, passives and upgrades apply. No Streamlit imports,
so they can be unit-tested.

units.json conventions
  unit["toggles"]  : [{"name": "Siege Mode", "label": "...", "default": false, "size_when_on": 3}]
  weapon["requires"]: {"Siege Mode": true}   -> weapon only offered when the toggle matches
  unit["options"]  : passives / upgrades / abilities that change attacks or defence:
      name, kind ("passive" | "upgrade" | "ability"), side ("attack" | "defend"),
      effects {...}                      (keys of AttackMods / DefendMods)
      phase     "any" | "ranged" | "melee"   attack phase of the weapon the option works with
      weapons   [names]                  only for weapons whose name contains one of these
      when      {"target_size_min": 3, "target_tag": "Armoured", "toggle": {"Siege Mode": false}}
      situational true                   needs a board condition the app cannot see (off by default)
      auto      true                     no checkbox: applied automatically whenever 'when' is met
      note      free text
  effects may contain "dmg_set_target_size": true  -> damage equals the target's Size (min 1)
"""
from __future__ import annotations

from engine import norm_tag

KIND_LABEL = {"passive": "Passive", "upgrade": "Upgrade", "ability": "Ability"}


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
