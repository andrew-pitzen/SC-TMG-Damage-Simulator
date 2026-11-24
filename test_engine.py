"""Sanity checks for engine.py, rules.py and the unit/card data. Run: python test_engine.py"""
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

import rules
from engine import AttackMods, DefendMods, Defender, Weapon, combine, simulate, simulate_attacks

N = 200_000
HERE = Path(__file__).parent

# 1) Published example: 2 models, RoA 4, Hit 3+, Armour 4+, Dmg 2 -> expected 5.3333 damage
r = simulate(Weapon(models=2, roa=4, hit=3, dmg=2), Defender(armour=4, models=50, hp=50), n=N, seed=1)
print("example damage mean:", r["damage"].mean(), "(expected 5.3333)")
assert abs(r["damage"].mean() - 5.3333) < 0.05

# 2) Surge: D6 surge vs matching tag (multi-tag Surge Type), ignored otherwise
w = Weapon(models=4, roa=3, hit=2, dmg=1, surge_die="D6", surge_tags=["Light", "Armoured"])
r = simulate(w, Defender(tags=["Armoured"], armour=6, models=50, hp=50), n=N, seed=2)
assert abs(r["hits"].mean() - 10.0) < 0.05
assert simulate(w, Defender(tags=["Biological"], armour=6, models=50, hp=50), n=1000, seed=2)["surged"].sum() == 0

# 3) D3+1 raises surge by exactly 1
w3 = Weapon(models=10, roa=3, hit=2, surge_die="D3", surge_tags=["Light"])
w31 = Weapon(models=10, roa=3, hit=2, surge_die="D3", surge_plus=1, surge_tags=["Light"])
d = Defender(tags=["Light"], armour=6, models=50, hp=50)
a, b = simulate(w3, d, n=N, seed=3)["surged"].mean(), simulate(w31, d, n=N, seed=3)["surged"].mean()
assert abs(a - 2.0) < 0.05 and abs(b - 3.0) < 0.05

# 4) Precision, Critical, Dodge, Tough
d = Defender(armour=6, models=50, hp=50)
base = simulate(Weapon(2, 3, 5), d, n=N, seed=4)["hits"].mean()
assert simulate(Weapon(2, 3, 5, precision=1), d, n=N, seed=4)["hits"].mean() > base + 0.5
crit = simulate(Weapon(2, 3, 3, critical=2), d, n=N, seed=5)
assert (crit["surged"] == np.minimum(2, crit["hits"])).all()
assert simulate(Weapon(2, 3, 3, critical=2), d, None, DefendMods(dodge=2), n=N, seed=5)["surged"].sum() == 0
no_t = simulate(Weapon(2, 3, 3), d, n=N, seed=6)["armour_fail"].mean()
tough = simulate(Weapon(2, 3, 3), d, None, DefendMods(tough=1), n=N, seed=6)["armour_fail"].mean()
assert tough < no_t

# 5) Evade only when eligible; Anti-Evade worsens target; grant_evade_roll uses own evade
ev_d = dict(armour=None, evade=4, models=50, hp=50)
assert simulate(Weapon(3, 2, 3), Defender(**ev_d), n=N, seed=7)["evaded"].sum() == 0
ev = simulate(Weapon(3, 2, 3), Defender(**ev_d, evade_eligible=True), n=N, seed=7)
ae = simulate(Weapon(3, 2, 3, anti_evade=2), Defender(**ev_d, evade_eligible=True), n=N, seed=7)
assert ev["evaded"].mean() > ae["evaded"].mean() > 0 and ae["meta"]["evade_target"] == 6
ger = simulate(Weapon(3, 2, 3), Defender(**ev_d), None, DefendMods(grant_evade_roll=True), n=N, seed=7)
assert abs(ger["evaded"].mean() - ev["evaded"].mean()) < 0.05 and ger["meta"]["evade_target"] == 4

# 6) Casualties
r = simulate(Weapon(1, 0, 2), Defender(models=3, hp=2, existing_damage=5, armour=None), n=10, seed=8)
assert (r["kills"] == 2).all() and (r["leftover"] == 1).all()

# 7) Pierce
q = Weapon(4, 3, 3, 1, pierce_x=2, pierce_tag="Armoured")
assert simulate(q, Defender(tags=["Armoured"], armour=None, models=50, hp=50), n=10)["meta"]["per_die_damage"] == 2
assert simulate(q, Defender(tags=["Light"], armour=None, models=50, hp=50), n=10)["meta"]["per_die_damage"] == 1

# 8) New effects: dmg_set, armour_set (never overrides a null armour)
assert simulate(Weapon(1, 1, 2, 1), Defender(armour=None, models=9, hp=9), AttackMods(dmg_set=3), n=10)["meta"]["per_die_damage"] == 3
assert simulate(Weapon(1, 1, 2), Defender(armour=6, models=9, hp=9), None, DefendMods(armour_set=4), n=10)["meta"]["armour_target"] == 4
assert simulate(Weapon(1, 1, 2), Defender(armour=None, models=9, hp=9), None, DefendMods(armour_set=4), n=10)["meta"]["armour_target"] is None

# 8b) Blast Template: S Dice "BT" moves exactly BT dice (bounded by hits)
bw = Weapon(1, 7, 2, 1, surge_die="BT", surge_tags=["Light"], bt=3)
br = simulate(bw, Defender(tags=["Light"], armour=6, models=50, hp=50), n=2000, seed=13)
assert (br["surged"] == np.minimum(3, br["hits"])).all() and br["meta"]["surge_die"] == "BT=3"
assert simulate(Weapon(1, 7, 2, 1, surge_die="BT", surge_tags=["Light"], bt=0),
                Defender(tags=["Light"], armour=6, models=50, hp=50), n=500, seed=13)["surged"].sum() == 0

# 9) Units: toggles, conditions and options
units = {k: v for k, v in json.loads((HERE / "units.json").read_text()).items() if not k.startswith("_")}
cards = json.loads((HERE / "cards.json").read_text())
find = lambda name: next(u for f in units.values() for n, u in f.items() if n == name)
opt = lambda unit, name: next(o for o in unit["options"] if o["name"] == name)

tank = find("Siege Tank")
off, on = {"Siege Mode": False}, {"Siege Mode": True}
assert rules.unit_size(tank, off) == 2 and rules.unit_size(tank, on) == 3
names = lambda s: [w["name"] for w in tank["weapons"] if rules.toggles_match(w.get("requires"), s)]
assert "Twin Cannon" in names(off) and "Shock Cannon" not in names(off)
assert "Shock Cannon" in names(on) and "Twin Cannon" not in names(on)
hp_ = opt(tank, "Heavy Plating")
assert rules.when_met(hp_, off, 2, []) and not rules.when_met(hp_, on, 3, [])
asr = opt(tank, "Aftershock Rounds")
assert rules.when_met(asr, on, 3, []) and not rules.when_met(asr, off, 2, [])
assert rules.option_effects(asr, 3) == {"dmg_set": 3} and rules.option_effects(asr, 0) == {"dmg_set": 1}

praetor = find("Khalai Praetor Guard")
tk = opt(praetor, "Titan Killers")
assert not rules.when_met(tk, {}, 2, []) and rules.when_met(tk, {}, 3, []) and tk.get("auto")
shock = next(w for w in tank["weapons"] if w["name"] == "Shock Cannon")
assert shock["blast"] and shock["surge"] == "BT" and shock["roa"] == 4 and shock["dmg"] is not None
magrail = opt(find("Spec Ops Marauder"), "Magrail Munitions")
assert combine([{"pierce_x": 2}, magrail["effects"]], AttackMods).pierce_x == 3   # keywords take the highest, not the sum
psi = praetor["weapons"][0]
pw = Weapon(3, psi["roa"], psi["hit"], psi["dmg"], psi["surge"], 0, psi["surge_tags"])
big = Defender(tags=["Armoured"], armour=None, models=50, hp=50)
dmg2 = simulate(pw, big, combine([tk["effects"]], AttackMods), n=N, seed=9)["damage"].mean()
dmg1 = simulate(pw, big, n=N, seed=9)["damage"].mean()
assert abs(dmg2 / dmg1 - 2.0) < 0.01, (dmg1, dmg2)

hydra = find("Hydralisk")
tough_up = opt(hydra, "Ancillary Carapace")
rr = Weapon(3, 3, 3)
a0 = simulate(rr, Defender(armour=5, models=50, hp=50), n=N, seed=10)["damage"].mean()
a1 = simulate(rr, Defender(armour=5, models=50, hp=50), None, combine([tough_up["effects"]], DefendMods), n=N, seed=10)["damage"].mean()
assert a1 < a0
zl = find("Zergling")
adr = opt(zl, "Adrenal Glands")
claws = zl["weapons"][0]
assert rules.matches_weapon(adr, claws) and not rules.matches_weapon(adr, {"name": "Spines", "phase": "ranged"})
assert simulate(Weapon(12, 2, 4), Defender(armour=6, models=50, hp=50), combine([adr["effects"]], AttackMods), n=N, seed=11)["hits"].mean() > \
       simulate(Weapon(12, 2, 4), Defender(armour=6, models=50, hp=50), n=N, seed=11)["hits"].mean() + 1.9
roach = find("Roach")
bile = opt(roach, "Hydriodic Bile")
acid = next(w for w in roach["weapons"] if w["name"] == "Acid Saliva")
aw = Weapon(3, acid["roa"], acid["hit"], acid["dmg"])
lt = Defender(tags=["Light"], armour=6, models=50, hp=50)
assert simulate(aw, lt, combine([bile["effects"]], AttackMods), n=1000, seed=12)["surged"].sum() > 0
assert simulate(aw, Defender(tags=["Armoured"], armour=6, models=50, hp=50), combine([bile["effects"]], AttackMods), n=1000, seed=12)["surged"].sum() == 0
# Shredding Claws: Claws already surge vs Light; the option adds Armoured (and keeps the Light surge)
for zname in ("Zergling", "Raptor (Zergling)"):
    zu = find(zname)
    shred = opt(zu, "Shredding Claws")
    zc = zu["weapons"][0]
    zw = Weapon(12, zc["roa"], zc["hit"], zc["dmg"], zc["surge"], 0, zc["surge_tags"])
    shred_m = combine([shred["effects"]], AttackMods)
    arm_d = Defender(tags=["Armoured"], armour=6, models=50, hp=50)
    lig_d = Defender(tags=["Light"], armour=6, models=50, hp=50)
    assert simulate(zw, arm_d, n=1000, seed=30)["surged"].sum() == 0
    assert simulate(zw, arm_d, shred_m, n=1000, seed=30)["surged"].sum() > 0
    assert simulate(zw, lig_d, shred_m, n=1000, seed=30)["surged"].sum() > 0
    assert simulate(zw, arm_d, shred_m, n=1000, seed=30)["meta"]["surge_die"] == "D3"

# 10) Data integrity: every option uses known effect keys; every weapon of every unit simulates
atk_keys = {f.name for f in fields(AttackMods)} | {"dmg_set_target_size"}
def_keys = {f.name for f in fields(DefendMods)}
count = opts_n = 0
for fac in units.values():
    for uname, u in fac.items():
        for o in u.get("options", []):
            assert o["kind"] in rules.KIND_LABEL and o["side"] in ("attack", "defend"), (uname, o["name"])
            assert set(o["effects"]) <= (atk_keys if o["side"] == "attack" else def_keys), (uname, o["name"])
            opts_n += 1
        for t in u.get("toggles", []):
            assert {"name", "label"} <= set(t)
        for w in u["weapons"]:
            for k in w.get("requires", {}):
                assert k in {t["name"] for t in u.get("toggles", [])}, (uname, w["name"])
all_atk = [o for f in units.values() for u in f.values() for o in u.get("options", []) if o["side"] == "attack"]
all_def = [o for f in units.values() for u in f.values() for o in u.get("options", []) if o["side"] == "defend"]
am = combine([rules.option_effects(o, 3) for o in all_atk] + [c["effects"] for c in cards["attacker"].values()], AttackMods)
dm = combine([o["effects"] for o in all_def] + [c["effects"] for c in cards["defender"].values()], DefendMods)
for fac in units.values():
    for u in fac.values():
        for w in u["weapons"]:
            wp = Weapon(u["models_default"], w["roa"] if w["roa"] is not None else 1, w["hit"],
                        w["dmg"] if w["dmg"] is not None else 1, w.get("surge", "None"), w.get("surge_plus", 0),
                        w.get("surge_tags", []), w.get("pierce_x", 0), w.get("pierce_tag", ""),
                        w.get("precision", 0), w.get("critical", 0), w.get("anti_evade", 0), 3 if w.get("blast") else 0)
            for fac2 in units.values():
                for t in fac2.values():
                    df = Defender(t["tags"], t["armour"], t["evade"], True, t["hp"], t["shield"], t["models_default"])
                    out = simulate(wp, df, am, dm, n=200, seed=0)
                    assert out["damage"].min() >= 0 and out["kills"].max() <= t["models_default"]
                    count += 1
for side_, keys_ in (("attacker", atk_keys), ("defender", def_keys)):
    for cname, c in cards[side_].items():
        assert c.get("faction", "Generic") in ("Terran", "Zerg", "Protoss", "Generic"), cname
        other_ = def_keys if side_ == "attacker" else atk_keys
        assert set(c["effects"]) <= (other_ if rules.is_opponent_card(c) else keys_), cname
        assert set(c.get("when", {})) <= {"own_size"}, cname
print(f"{opts_n} unit options validated; data smoke test: {count} weapon-vs-unit simulations OK")

# 11) Option keywords: top-level "keywords" on an option are merged with the weapon's keywords
for fac in units.values():
    for uname, u in fac.items():
        for o in u.get("options", []):
            assert "keywords" not in o["effects"], (uname, o["name"], "keywords belong next to effects, not inside")
            assert all(isinstance(k, str) for k in o.get("keywords", [])), (uname, o["name"])
nova = find("Nova (Stealth Mode)")
rifle, headshot = nova["weapons"][0], opt(nova, "Head Shot")
assert rules.find_keyword(rules.weapon_keywords(rifle, []), "concentrated fire") is None
merged = rules.weapon_keywords(rifle, [headshot])
assert rules.keyword_value(merged, "concentrated fire", min) == 1
assert rules.keyword_value(merged, "long range") == 0 and rules.find_keyword(merged, "long range")  # presence, no number
assert rules.keyword_value(['Burst Fire 4" (3)'], "burst fire") == 3 and rules.keyword_value(['Burst Fire 8" (2)'], "burst fire") == 2
assert rules.keyword_value(["Concentrated Fire (3)", "Concentrated Fire (1)"], "concentrated fire", min) == 1
assert rules.weapon_keywords(None, []) == []
assert rules.keyword_value(find("Nova (Assault Mode)")["weapons"][0]["keywords"], "burst fire") == 3

# 12) Concentrated Fire (X): at most X models destroyed, excess damage (incl. partial) ignored
cf_def = Defender(armour=None, models=20, hp=2)
free = simulate(Weapon(10, 3, 2, 2), cf_def, n=2000, seed=14)
assert free["kills"].max() > 1
cf1 = simulate(Weapon(10, 3, 2, 2, concentrated_fire=1), cf_def, n=2000, seed=14)
assert (cf1["kills"] == 1).all() and (cf1["leftover"] == 0).all() and (cf1["damage"] == 2).all()
assert cf1["meta"]["concentrated_fire"] == 1
# the first model carries the Shield: killing 2 models (hp 2, first one +3 Shield) takes 5 + 2 = 7 damage
cf2 = simulate(Weapon(10, 3, 2, 2, concentrated_fire=2), Defender(armour=None, models=20, hp=2, shield=3), n=2000, seed=14)
assert (cf2["kills"] == 2).all() and (cf2["damage"] == 7).all() and (cf2["leftover"] == 0).all()
# not capped when the attack kills fewer models than X: results match the uncapped attack
weak_w, weak_d = Weapon(1, 2, 3, 1), Defender(armour=None, models=20, hp=2)
a_ = simulate(weak_w, weak_d, n=2000, seed=15)
b_ = simulate(Weapon(1, 2, 3, 1, concentrated_fire=5), weak_d, n=2000, seed=15)
assert (a_["kills"] == b_["kills"]).all() and (a_["damage"] == b_["damage"]).all() and (a_["leftover"] == b_["leftover"]).all()
# an option / card can grant it; the lowest value wins; combine() keeps the lowest non-zero value
assert combine([{"concentrated_fire": 3}, {"concentrated_fire": 1}, {"concentrated_fire": 0}], AttackMods).concentrated_fire == 1
via_mods = simulate(Weapon(10, 3, 2, 2), cf_def, AttackMods(concentrated_fire=1), n=2000, seed=14)
assert (via_mods["kills"] == 1).all()
both = simulate(Weapon(10, 3, 2, 2, concentrated_fire=3), cf_def, AttackMods(concentrated_fire=2), n=2000, seed=14)
assert (both["kills"] == 2).all()
# visible-model limit still applies on its own
vis = simulate(Weapon(10, 3, 2, 2, concentrated_fire=4), Defender(armour=None, models=20, hp=2, visible=3), n=500, seed=14)
assert (vis["kills"] == 3).all()
# real data: Immortal's Photon Disruptors carry Concentrated Fire (1)
immortal = find("Immortal")
assert all(rules.keyword_value(w.get("keywords", []), "concentrated fire") == 1
           for w in immortal["weapons"] if w["name"] in ("Left Photon Disruptor", "Right Photon Disruptor"))

# 13) Swap attacker/defender: faction and unit picks trade places
S = lambda **kw: dict(kw)
st_ = S(a_fac="Terran", d_fac="Zerg", a_unit_Terran="Marine", d_unit_Zerg="Roach")
assert rules.swap_selection(st_, units, "Custom (manual)") == []
assert (st_["a_fac"], st_["d_fac"], st_["a_unit_Zerg"], st_["d_unit_Terran"]) == ("Zerg", "Terran", "Roach", "Marine")
rules.swap_selection(st_, units, "Custom (manual)")   # swapping twice restores the picks
assert (st_["a_fac"], st_["d_fac"], st_["a_unit_Terran"], st_["d_unit_Zerg"]) == ("Terran", "Zerg", "Marine", "Roach")
st_ = S(a_fac="Zerg", d_fac="Zerg", a_unit_Zerg="Roach", d_unit_Zerg="Hydralisk")   # same faction on both sides
rules.swap_selection(st_, units, "Custom (manual)")
assert (st_["a_unit_Zerg"], st_["d_unit_Zerg"]) == ("Hydralisk", "Roach")
st_ = S(a_fac="Terran", d_fac="Protoss", a_unit_Terran="Marine", d_unit_Protoss="Pylon")   # Pylon has no weapons
notes = rules.swap_selection(st_, units, "Custom (manual)")
assert st_["a_fac"] == "Protoss" and "a_unit_Protoss" not in st_ and st_["d_unit_Terran"] == "Marine" and len(notes) == 1
st_ = S(a_fac="Custom (manual)", d_fac="Terran", d_unit_Terran="Marine")   # Custom on one side
rules.swap_selection(st_, units, "Custom (manual)")
assert (st_["a_fac"], st_["d_fac"], st_["a_unit_Terran"]) == ("Terran", "Custom (manual)", "Marine")

# 14) Main weapon + Sidearm in one attack: damage adds, casualties are worked out once
gd = Defender(armour=None, models=50, hp=50)
mw, sw = Weapon(2, 3, 3, 1, name="main"), Weapon(1, 4, 4, 2, name="side")
solo_m = simulate(mw, gd, n=N, seed=16)["damage"].mean()
solo_s = simulate(sw, gd, n=N, seed=17)["damage"].mean()
both = simulate_attacks([(mw, None), (sw, None)], gd, None, n=N, seed=18)
assert abs(both["damage"].mean() - (solo_m + solo_s)) < 0.1
assert [m["name"] for m in both["meta"]["weapons"]] == ["main", "side"] and both["meta"]["pool"] == 6
assert abs(both["attack_pool"].mean() - 10) < 1e-9          # 6 + 4 dice, stages are summed
# single-weapon call is unchanged
one = simulate_attacks([(mw, None)], gd, n=500, seed=19)
assert (one["damage"] == simulate(mw, gd, n=500, seed=19)["damage"]).all() and len(one["meta"]["weapons"]) == 1
# TOTAL damage reduction is taken off once for the whole attack, not once per weapon
td = Defender(armour=None, models=50, hp=50)
dm_ = DefendMods(total_reduction=2)
raw = simulate_attacks([(mw, None), (sw, None)], td, None, n=2000, seed=20)["damage"]
cut = simulate_attacks([(mw, None), (sw, None)], td, dm_, n=2000, seed=20)["damage"]
assert (cut == np.maximum(0, raw - 2)).all()
# the first model carries the Shield once; leftover damage is shared across weapons
sh = simulate_attacks([(Weapon(1, 0, 2, 1), None), (Weapon(1, 0, 2, 1), None)],
                      Defender(models=3, hp=2, existing_damage=5, armour=None), n=10, seed=21)
assert (sh["kills"] == 2).all() and (sh["leftover"] == 1).all()
# Concentrated Fire limits only the weapon that has it, not the other weapons in the attack
cfd = Defender(armour=None, models=20, hp=2)
cf_w, free_w = Weapon(10, 3, 2, 2, concentrated_fire=1, name="cf"), Weapon(10, 3, 2, 2, name="free")
cf_first = simulate_attacks([(cf_w, None), (free_w, None)], cfd, n=2000, seed=22)
only_free = simulate_attacks([(free_w, None)], cfd, n=2000, seed=22)
assert cf_first["kills"].min() > 1 and cf_first["kills"].mean() > 8      # sidearm is NOT capped to 1
assert cf_first["meta"]["weapons"][0]["concentrated_fire"] == 1 and cf_first["meta"]["weapons"][1]["concentrated_fire"] == 0
free_first = simulate_attacks([(free_w, None), (cf_w, None)], cfd, n=2000, seed=22)
assert (free_first["kills"] >= np.minimum(only_free["kills"] + 1, 20)).mean() > 0.9  # cf weapon adds at most 1 more
assert (free_first["kills"] <= only_free["kills"] + 1).all()
# two Concentrated Fire weapons: each destroys at most its own X
two_cf = simulate_attacks([(cf_w, None), (Weapon(10, 3, 2, 2, concentrated_fire=2), None)], cfd, n=2000, seed=24)
assert (two_cf["kills"] == 3).all()
# real data: every unit's Sidearm weapon can be fired together with a non-sidearm ranged weapon
sidearm_units = 0
for fac in units.values():
    for uname, u in fac.items():
        rng_w = [w for w in u["weapons"] if w["phase"] == "ranged"]
        sides = [w for w in rng_w if rules.find_keyword(w.get("keywords", []), "sidearm")]
        mains = [w for w in rng_w if w not in sides]
        if sides and mains:
            sidearm_units += 1
            mk = lambda w: Weapon(u["models_default"], w["roa"], w["hit"], w["dmg"], w.get("surge", "None"),
                                  w.get("surge_plus", 0), w.get("surge_tags", []), w.get("pierce_x", 0),
                                  w.get("pierce_tag", ""), name=w["name"])
            out = simulate_attacks([(mk(mains[0]), None)] + [(mk(x), None) for x in sides],
                                   Defender(["Armoured", "Light"], 4, 5, True, 5, 0, 10), n=300, seed=23)
            assert len(out["meta"]["weapons"]) == 1 + len(sides) and out["damage"].min() >= 0, uname
assert sidearm_units >= 3, sidearm_units
print(f"sidearm combos checked on {sidearm_units} units")

# 15) Scope: defender effects can apply to the main weapon only, or to every weapon
tg = Defender(armour=5, models=50, hp=50)
w1, w2 = Weapon(4, 3, 3, 1, name="main"), Weapon(4, 3, 3, 1, name="side")
items = [("first", {"tough": 2}), ("all", {"dodge": 1}), ("first", {"total_reduction": 3, "hp_bonus": 1})]
shared, main_m, other_m = rules.defend_mods_by_scope(items)
assert (main_m.tough, main_m.dodge) == (2, 1) and (other_m.tough, other_m.dodge) == (0, 1)
assert shared.tough == 0 and shared.dodge == 1 and shared.total_reduction == 3 and shared.hp_bonus == 1  # attack-wide keys
assert other_m.total_reduction == 0
r = simulate_attacks([(w1, None, DefendMods(tough=2)), (w2, None, DefendMods())], tg, DefendMods(), n=2000, seed=40)
assert r["meta"]["weapons"][0]["tough"] == 2 and r["meta"]["weapons"][1]["tough"] == 0
# omitting the third item keeps the old behaviour (shared mods for every weapon)
r = simulate_attacks([(w1, None), (w2, None)], tg, DefendMods(tough=1), n=2000, seed=40)
assert [m["tough"] for m in r["meta"]["weapons"]] == [1, 1]
# data: scope tags
assert rules.is_all_scope(opt(tank, "Heavy Plating")) and rules.is_all_scope(opt(tank, "Aftershock Rounds"))
assert rules.is_all_scope(cards["defender"]["Zerg: Zagara - Tough (1) for Size 3 units"])
assert not rules.is_all_scope(cards["defender"]["Zerg: Evolution Chamber - Carapace (Tough 1)"])
assert any("Adept Shade" in n for n in cards["attacker"]) and all(
    c.get("scope", "first") in ("first", "all") for side_ in cards.values() if isinstance(side_, dict) for c in side_.values() if isinstance(c, dict))

# 16) Impact: separate extra attack, no attacker mods, only the defender's all-scope effects
zeal, zer = find("Zealot"), find("Zeratul")
assert zeal["impact"]["per_model"] and zer["impact"]["hit"] == 4
up = next(o for o in zeal["options"] if rules.is_impact_option(o))
assert up["impact_dice"] >= 1 and not rules.matches_weapon(up, zeal["weapons"][0])   # not listed under the weapon
imp = Weapon(1, 6, 4, 1, name="Impact")
iw = simulate_attacks([(Weapon(3, 4, 3, 1, name="blades"), AttackMods(precision=3, critical=2), DefendMods(tough=5)),
                       (imp, AttackMods(), DefendMods())], tg, DefendMods(), n=N, seed=41)
only_imp = simulate_attacks([(imp, AttackMods(), DefendMods())], tg, DefendMods(), n=N, seed=42)
m_imp = iw["meta"]["weapons"][1]
assert m_imp["name"] == "Impact" and m_imp["precision"] == 0 and m_imp["critical"] == 0 and m_imp["tough"] == 0 and m_imp["pool"] == 6
assert m_imp["hit_target"] == 4 and abs(iw["meta"]["weapons"][1]["pool"] - 6) == 0
# impact alone: 6 dice, hit 4+ (p=.5), armour 5+ (p fail = 4/6) -> 6*.5*(4/6)=2.0 damage
assert abs(only_imp["damage"].mean() - 2.0) < 0.05, only_imp["damage"].mean()
# an all-scope defender effect (Tough 1) lowers impact damage
tough_imp = simulate_attacks([(imp, AttackMods(), DefendMods(tough=1))], tg, DefendMods(), n=N, seed=42)["damage"].mean()
assert tough_imp < only_imp["damage"].mean() - 0.3
# damage of the pooled attack = main + impact (same seed stream order differs, so compare means)
main_only = simulate_attacks([(Weapon(3, 4, 3, 1, name="blades"), AttackMods(precision=3, critical=2), DefendMods(tough=5))],
                             tg, DefendMods(), n=N, seed=43)["damage"].mean()
assert abs(iw["damage"].mean() - (main_only + only_imp["damage"].mean())) < 0.1
print("all checks passed")

# 17) Per-attack targeting of defender cards: main weapon, Sidearm and Impact can each get their own cards
items = [("first", {"tough": 2}, ["main"]),
         ("first", {"dodge": 1, "total_reduction": 3}, ["sa:side", "impact"]),
         ("all", {"evade_mod": 1}, [])]
shared, per = rules.defend_mods_by_target(items, ["main", "sa:side", "impact"])
assert (per["main"].tough, per["main"].dodge) == (2, 0)
assert (per["sa:side"].tough, per["sa:side"].dodge) == (0, 1)
assert (per["impact"].tough, per["impact"].dodge) == (0, 1)
assert all(m.evade_mod == 1 for m in per.values())
assert shared.total_reduction == 3 and shared.evade_mod == 1 and shared.tough == 0
shared, per = rules.defend_mods_by_target([("first", {"tough": 1, "hp_bonus": 1}, [])], ["main"])
assert per["main"].tough == 0 and shared.hp_bonus == 1
print("all checks passed (targeting)")

# 18) Auto-hit: every die hits, no Hit roll; Zeratul data
ah = simulate(Weapon(1, 4, 6, 1, auto_hit=True), Defender(armour=None, models=50, hp=50), n=500, seed=50)
assert (ah["hits"] == 4).all() and ah["meta"]["hit_target"] is None and (ah["damage"] == 4).all()
zer_u = find("Zeratul")
assert zer_u["impact"]["hit"] == 4 and zer_u["impact"]["dice"] == 4
assert rules.auto_hit_list(zer_u)[0]["dice"] == 4 and rules.phase_ok(rules.auto_hit_list(zer_u)[0], "melee") and not rules.phase_ok(rules.auto_hit_list(zer_u)[0], "ranged")
# defender cards can be pointed at the auto-hit attack alone
sh_, per_ = rules.defend_mods_by_target([("first", {"tough": 2}, ["auto"])], ["main", "impact", "auto"])
assert (per_["auto"].tough, per_["main"].tough, per_["impact"].tough) == (2, 0, 0)
print("all checks passed (auto-hit)")

assert rules.auto_hit_list({}) == [] and len(rules.auto_hit_list({"auto_hits": [{"label": "A"}, {"label": "B"}]})) == 2
assert rules.auto_hit_list({"auto_hits": {"label": "A"}}) == [{"label": "A"}]
print("all checks passed (auto-hit list)")

# 19. Impact-only attacker cards
import json as _j
_cards = _j.load(open("cards.json"))["attacker"]
_imp = [n for n, c in _cards.items() if rules.card_targets(c) == ["impact"]]
assert len(_imp) >= 2 and all(_cards[n]["effects"].get("hit_mod") == 1 for n in _imp)
assert rules.card_label(_imp[0], _cards[_imp[0]]).endswith("[Impact only]")
from engine import Weapon as _W, Defender as _D, simulate_attacks as _sim, AttackMods as _AM, DefendMods as _DM
_d = _D(["Biological"], 4, 0, False, 3, 0, 10, 10, 10)
_w = _W(1, 40, 4, 1, name="Impact")
_a = _sim([(_w, _AM(), _DM())], _d, _DM(), n=4000, seed=1)["damage"].mean()
_b = _sim([(_w, _AM(hit_mod=1), _DM())], _d, _DM(), n=4000, seed=1)["damage"].mean()
assert _b > _a, (_a, _b)
print("all checks passed (impact-only cards)")

# 22. Direct damage: no dice, ignores Armour, Evade, Total damage reduction and Concentrated Fire
dd_d = Defender(armour=2, evade=2, evade_eligible=True, models=10, hp=3, shield=0)
only_dd = simulate_attacks([(Weapon(1, 0, 2, 0), None)], dd_d, DefendMods(total_reduction=5, tough=3), n=500, seed=60, direct_damage=2)
assert (only_dd["damage"] == 2).all() and only_dd["meta"]["direct_damage"] == 2 and (only_dd["kills"] == 0).all()
big_dd = simulate_attacks([(Weapon(1, 0, 2, 0), None)], dd_d, None, n=50, seed=60, direct_damage=7)
assert (big_dd["damage"] == 7).all() and (big_dd["kills"] == 2).all() and (big_dd["leftover"] == 1).all()
no_dd = simulate_attacks([(Weapon(2, 2, 3, 1), None)], dd_d, None, n=2000, seed=61)["damage"]
with_dd = simulate_attacks([(Weapon(2, 2, 3, 1), None)], dd_d, None, n=2000, seed=61, direct_damage=3)["damage"]
assert (with_dd == no_dd + 3).all()
stk = find("Stukov")
assert rules.direct_damage_list(stk)[0]["dmg"] == 2 and rules.direct_damage_list({}) == []
assert rules.direct_damage_list({"direct_damage": {"label": "x", "dmg": 1}}) == [{"label": "x", "dmg": 1}]
print("all checks passed (direct damage)")

# 23. Debuff cards switch sides but still change the right numbers
bile_n = "Zerg: Ravager Potent Bile (Armour -1 debuff on enemy)"
sal_n = "Zerg: Vile Debilitating Saliva (Hit -1 debuff on attacker)"
assert bile_n in cards["attacker"] and sal_n in cards["defender"]
assert bile_n not in cards["defender"] and sal_n not in cards["attacker"]
bile_c, sal_c = cards["attacker"][bile_n], cards["defender"][sal_n]
assert rules.is_opponent_card(bile_c) and rules.is_opponent_card(sal_c) and not rules.is_opponent_card(cards["attacker"]["Generic: +1 to Hit"])
assert combine([bile_c["effects"]], DefendMods).armour_mod == -1 and combine([sal_c["effects"]], AttackMods).hit_mod == -1
r_b = simulate(Weapon(2, 3, 3), Defender(armour=5, models=50, hp=50), None, combine([bile_c["effects"]], DefendMods), n=500, seed=70)
assert r_b["meta"]["armour_target"] == 6
r_s = simulate(Weapon(2, 3, 3), Defender(armour=5, models=50, hp=50), combine([sal_c["effects"]], AttackMods), None, n=500, seed=70)
assert r_s["meta"]["hit_target"] == 4
print("all checks passed (debuff cards)")

# 24. Phase tags are applied per attack; weapon checkboxes support mixed weapons
tphase = {"w:gun": "ranged", "w:blade": "melee", "impact": rules.IMPACT_PHASE, "ab:x": "any"}
ids = list(tphase)
bile_n = "Zerg: Ravager Potent Bile (Armour -1 debuff on enemy)"
assert cards["attacker"][bile_n]["phase"] == "melee"
raw = [(rules.scope_of(cards["attacker"][bile_n]), cards["attacker"][bile_n]["effects"], [], cards["attacker"][bile_n])]
_, per = rules.defend_mods_by_target(rules.phase_filtered_items(raw, ids, tphase), ids)
assert per["w:gun"].armour_mod == 0 and per["w:blade"].armour_mod == -1 and per["impact"].armour_mod == 0   # Impact is ranged
assert all(cards["attacker"][n]["phase"] == rules.IMPACT_PHASE for n, c_ in cards["attacker"].items() if rules.card_targets(c_) == ["impact"])
assert per["ab:x"].armour_mod == -1          # a phase-less ability attack is not excluded
# a first-scope card only reaches its assigned attacks that match its phase; nothing matching -> dropped
f = rules.phase_filtered_items([("first", {"tough": 1, "total_reduction": 2}, ["w:gun"], {"phase": "melee"})], ids, tphase)
assert f == []
sh_, _ = rules.defend_mods_by_target(f, ids)
assert sh_.total_reduction == 0
f = rules.phase_filtered_items([("first", {"tough": 1}, ["w:gun", "w:blade"], {"phase": "ranged"})], ids, tphase)
assert f == [("first", {"tough": 1}, ["w:gun"])]
assert rules.phase_filtered_items([("all", {"dodge": 1}, [], {})], ids, tphase) == [("all", {"dodge": 1}, [])]
# default target = first attack that fits the card's phase
assert rules.default_target({"phase": "melee"}, ids, tphase) == "w:blade"
assert rules.default_target({"phase": "ranged"}, ids, tphase) == "w:gun"
assert rules.default_target({}, ids, tphase) == "w:gun" and rules.default_target({}, [], tphase) is None
# which abilities are offered: everything with no weapon ticked, else only matching phases
P = lambda ph, kind="passive": {"phase": ph, "kind": kind}
assert rules.phase_offered(P("melee"), set()) and rules.phase_offered(P("ranged"), set())
assert not rules.phase_offered(P("melee"), {"ranged"}) and rules.phase_offered(P("any"), {"ranged"})
assert rules.phase_offered(P("melee", "ability"), {"ranged"}) and rules.phase_offered({"phase": "melee"}, {"ranged"})   # abilities / attacks always show
assert rules.reaches_any({"phase": "melee"}, tphase) and not rules.reaches_any({"phase": "melee"}, {"w:gun": "ranged"})
assert "[melee only]" in rules.card_label(bile_n, cards["attacker"][bile_n]) and "[all weapons]" in rules.card_label(bile_n, cards["attacker"][bile_n])
assert rules.card_label("x", {}) == "x"
# melee + ranged weapons in one attack, each with its own defender mods: only the melee one meets Armour -1
mel, rng_ = Weapon(2, 4, 3, 1, name="blade"), Weapon(2, 4, 3, 1, name="gun")
r = simulate_attacks([(rng_, None, per["w:gun"]), (mel, None, per["w:blade"])], Defender(armour=5, models=50, hp=50), n=300, seed=80)
assert [m["armour_target"] for m in r["meta"]["weapons"]] == [5, 6]
# ability attacks in the real data are well-formed
for fac in units.values():
    for uname, u in fac.items():
        for ab in rules.ability_attack_list(u):
            assert ab.get("phase", "any") in ("any", "ranged", "melee") and ab.get("dice", 1) >= 0, uname
# Marine's Combat Shield: melee attacks + Impact + auto-hit abilities, but not ranged weapons / ability attacks
cs_o = next(o for o in find("Marine")["options"] if o["name"] == "Combat Shield")
tp2 = {"w:rifle": "ranged", "w:strike": "melee", "impact": "ranged", "auto:Shadow Strike": "melee",
       "auto:Bile": "ranged", "ab:Auto Turret": "ranged"}
ids2 = list(tp2)
_, per2 = rules.defend_mods_by_target(rules.phase_filtered_items(
    [(rules.scope_of(cs_o), cs_o["effects"], ["w:strike"], cs_o)], ids2, tp2), ids2)
assert {t_: m.grant_evade_roll for t_, m in per2.items()} == {"w:rifle": False, "w:strike": True, "impact": True,
        "auto:Shadow Strike": True, "auto:Bile": True, "ab:Auto Turret": False}
assert rules.reaches_any(cs_o, {"impact": "ranged"}) and not rules.reaches_any(cs_o, {"w:rifle": "ranged"})
print("all checks passed (per-attack phases)")
