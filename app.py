"""StarCraft TMG attack Monte Carlo calculator (Streamlit).

Run:  streamlit run app.py
Unit stats, toggles, passives and upgrades live in units.json; tactical cards in cards.json.
Both are plain JSON you can edit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rules
from engine import AttackMods, DefendMods, Defender, Weapon, combine, simulate

N_SIMS = 20_000
HERE = Path(__file__).parent
COMBAT_TAGS = ["Ground", "Flying", "Armoured", "Biological", "Light", "Mechanical", "Psionic"]
CARDS = json.loads((HERE / "cards.json").read_text())
UNITS = {k: v for k, v in json.loads((HERE / "units.json").read_text()).items() if not k.startswith("_")}
FACTIONS = list(UNITS)
CUSTOM = "Custom (manual)"

st.set_page_config(page_title="StarCraft TMG Attack Simulator", layout="wide")
st.title("StarCraft TMG - Attack Simulator")
st.caption(f"Monte Carlo: {N_SIMS:,} simulated attacks. Hit -> Surge -> Armour -> Evade -> Damage -> Casualties.")


# ------------------------------- helpers ----------------------------------- #
def tn(v):
    return "-" if v is None else f"{v}+"


def weapon_label(w):
    roa = "?" if w.get("roa") is None else (f"BT+{w['roa']}" if w.get("blast") else w["roa"])
    dmg = "?" if w.get("dmg") is None else w["dmg"]
    surge = ""
    if w.get("surge", "None") != "None":
        die = w["surge"] + (f"+{w['surge_plus']}" if w.get("surge_plus") else "")
        surge = f", S {die} ({'/'.join(w.get('surge_tags', []))})"
    kws = f", {', '.join(w['keywords'])}" if w.get("keywords") else ""
    return (f"{w['name']} [{'ranged' if w['phase'] == 'ranged' else 'melee'}] - "
            f"RoA {roa}, Hit {w['hit']}+, DMG {dmg}{surge}{kws}")


def target_input(label, key, default, allow_null=False):
    options = [2, 3, 4, 5, 6] + (["-"] if allow_null else [])
    default = "-" if default is None else default
    val = st.selectbox(label, options, index=options.index(default), key=key,
                       format_func=lambda v: v if v == "-" else f"{v}+")
    return None if val == "-" else val


def cards_for(side, faction):
    """Cards of the selected faction plus Generic ones (all cards for a Custom unit)."""
    return [n for n, c in CARDS[side].items()
            if faction == CUSTOM or c.get("faction", "Generic") in (faction, "Generic")]


def state_key(state):
    return "".join(str(int(v)) for v in state.values())


def toggle_inputs(unit, prefix):
    state = {}
    for t in unit.get("toggles", []):
        state[t["name"]] = st.checkbox(t.get("label", t["name"]), value=t.get("default", False),
                                       key=f"{prefix}_tg_{t['name']}")
    return state


def option_inputs(options, prefix, own_state, target_size, target_tags):
    """Render a checkbox per option. Returns the options that are ticked and whose condition is met."""
    chosen = []
    if options:
        st.markdown("**Unit passives & upgrades**")
    for o in options:
        met = rules.when_met(o, own_state, target_size, target_tags)
        if o.get("auto"):
            if met:
                chosen.append(o)
            st.caption(f"{rules.option_label(o)}: " + ("applied automatically" if met else "not active (condition not met)"))
            continue
        label = rules.option_label(o) + ("" if met else " - inactive (condition not met)")
        on = st.checkbox(label, value=rules.default_checked(o), key=f"{prefix}_opt_{o['name']}",
                         disabled=not met, help=o.get("note"))
        if on and met:
            chosen.append(o)
    return chosen


def swap_sides():
    """Button callback: swap the faction and unit picks of the attacker and the defender."""
    st.session_state["swap_notes"] = rules.swap_selection(st.session_state, UNITS, CUSTOM)


st.button("⇄ Swap attacker and defender", on_click=swap_sides,
          help="Swaps the faction and unit selections between the two sides. Other settings (toggles, "
               "model counts, tweaks, cards) stay on the side they were set on.")
for _note in st.session_state.pop("swap_notes", []):
    st.info(_note)

# The defender block runs first: attacker conditions (e.g. Titan Killers) depend on the target's Size and tags.
left, right = st.columns(2)

# ------------------------------- Defender ---------------------------------- #
with right:
    st.subheader("Defender")
    d_faction = st.selectbox("Faction", FACTIONS + [CUSTOM], key="d_fac")
    if d_faction == CUSTOM:
        d_unit = "custom"
        ud = {"tags": ["Ground", "Biological", "Light"], "armour": 5, "evade": 5, "hp": 3, "shield": 0,
              "models_default": 5, "models_max": 60, "size": 2, "toggles": [], "options": []}
    else:
        d_unit = st.selectbox("Unit", list(UNITS[d_faction]), key=f"d_unit_{d_faction}")
        ud = UNITS[d_faction][d_unit]
    k = f"{d_faction}_{d_unit}"
    d_state = toggle_inputs(ud, f"d_{k}")
    d_tags = st.multiselect("Combat Tags", COMBAT_TAGS + ["Unique"],
                            default=[t for t in ud["tags"] if t in COMBAT_TAGS], key=f"d_tags_{k}")
    evade_on = st.checkbox("Evade roll eligible", key=f"d_evon_{k}",
                           help="Target is Engaged, on High Ground, or an ability grants a roll. Evade is only rolled in those cases.")
    with st.expander("Stats: Armour / Evade / HP / Shield"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            armour = target_input("Armour", f"d_arm_{k}", ud["armour"], allow_null=True)
        with c2:
            evade = target_input("Evade", f"d_eva_{k}", ud["evade"], allow_null=True)
        d_hp = c3.number_input("HP / model", 1, 60, ud["hp"], key=f"d_hp_{k}")
        d_shield = c4.number_input("Shield", 0, 60, ud["shield"], key=f"d_sh_{k}",
                                   help="Added to the first model's HP.")
    with st.expander("Unit: Size / Models / Visible / Damage"):
        c1, c2, c3, c4 = st.columns(4)
        d_size = c1.number_input("Size", 0, 6, rules.unit_size(ud, d_state), key=f"d_size_{k}_{state_key(d_state)}",
                                 help="Used by size-based rules such as Titan Killers, Aftershock Rounds or Zagara's buffs.")
        d_models = c2.number_input("Models", 1, max(60, ud["models_max"]), ud["models_default"], key=f"d_models_{k}")
        d_vis = c3.number_input("Visible", 1, int(d_models), int(d_models), key=f"d_vis_{k}_{d_models}",
                                help="Visible models. Casualties can never exceed this.")
        d_exist = c4.number_input("Damage on unit", 0, 200, 0, key=f"d_ex_{k}",
                                  help="Damage already on the unit.")
    st.caption(f"Armour {tn(armour)} - Evade {tn(evade)} - HP {d_hp} - Shield {d_shield} - Size {d_size} - "
               f"{d_models} models ({d_vis} visible) - {d_exist} damage on unit")

    d_opts = option_inputs([o for o in ud.get("options", []) if o["side"] == "defend"],
                           f"d_{k}", d_state, d_size, d_tags)

    st.markdown("**Tactical cards / other buffs**")
    def_cards = st.multiselect("Defender cards", cards_for("defender", d_faction), key=f"d_cards_{d_faction}")
    with st.expander("Custom defender modifiers"):
        c1, c2 = st.columns(2)
        cd_arm = c1.number_input("Armour modifier (+ = easier)", -4, 4, 0, key="cd_arm")
        cd_eva = c2.number_input("Evade modifier", -4, 4, 0, key="cd_eva")
        cd_ge = c1.selectbox("Grant Evade roll at", [0, 2, 3, 4, 5, 6], key="cd_ge",
                             format_func=lambda v: "none" if v == 0 else f"{v}+")
        cd_dr = c2.number_input("Damage reduction per die", 0, 5, 0, key="cd_dr")
        cd_rr = c1.selectbox("Re-roll armour", ["none", "ones", "fails"], key="cd_rr")
        cd_sh = c2.number_input("Extra Shield", 0, 20, 0, key="cd_sh")
        cd_to = c1.number_input("TOUGH (X)", 0, 10, 0, key="cd_to")
        cd_do = c2.number_input("DODGE (X)", 0, 10, 0, key="cd_do")
        cd_tr = c1.number_input("Total damage reduction", 0, 30, 0, key="cd_tr")
        cd_ap = c2.number_input("Attack dice removed", 0, 30, 0, key="cd_ap")
        cd_is = c1.checkbox("Ignore Surge", key="cd_is")

# ------------------------------- Attacker ---------------------------------- #
with left:
    st.subheader("Attacker")
    a_faction = st.selectbox("Faction", FACTIONS + [CUSTOM], key="a_fac")
    a_opts = []
    long_range = False
    burst_fire = False
    burst_dice = 3   # extra RoA per model when Burst Fire is on (the number in "Burst Fire X\" (N)")
    a_size = 2

    if a_faction == CUSTOM:
        a_phase = "ranged"
        c1, c2 = st.columns(2)
        models = c1.number_input("Models firing", 1, 60, 3, key="a_models")
        roa = c2.number_input("RoA (dice per model)", 0, 10, 2, key="a_roa")
        c1, c2 = st.columns(2)
        hit = target_input("Hit", "a_hit", 4)
        dmg = c2.number_input("DMG", 0, 10, 1, key="a_dmg")
        c1, c2, c3 = st.columns(3)
        surge_die = c1.selectbox("S Dice", ["None", "D3", "D6"], key="a_sd")
        surge_plus = c2.number_input("S Dice +", 0, 3, 0, key="a_sp", disabled=surge_die == "None")
        surge_tags = c3.multiselect("Surge Type", COMBAT_TAGS, key="a_st", disabled=surge_die == "None")
        c1, c2, c3 = st.columns(3)
        pierce_x = c1.number_input("PIERCE (X)", 0, 10, 0, key="a_px")
        pierce_tag = c2.selectbox("PIERCE vs tag", [""] + COMBAT_TAGS, key="a_pt", disabled=pierce_x == 0)
        a_phase = c3.selectbox("Attack type", ["ranged", "melee"], key="a_ph")
        c1, c2, c3 = st.columns(3)
        w_prec = c1.number_input("PRECISION (X)", 0, 10, 0, key="a_prec")
        w_crit = c2.number_input("CRITICAL HIT (X)", 0, 10, 0, key="a_crit")
        w_ae = c3.number_input("ANTI-EVADE (X)", 0, 4, 0, key="a_ae")
        long_range = st.checkbox("Long Range (target beyond normal range: -1 to hit)", key="a_lr",
                                 help="Makes the Hit roll 1 harder (e.g. 3+ becomes 4+).")
        c1, c2 = st.columns(2)
        burst_fire = c1.checkbox("Burst Fire (target within range: increase RoA)", key="a_bf",
                                 help="Increases the RoA of the weapon.")
        burst_dice = c2.number_input("Burst Fire bonus RoA (per model)", 1, 10, 3, key="a_bfd",
                                     disabled=not burst_fire)
        w_cf = st.number_input("CONCENTRATED FIRE (X, 0 = off)", 0, 10, 0, key="a_cf",
                               help="The attack can destroy at most X models; excess damage is ignored.")
        a_size = st.number_input("Attacker Size (for size-based cards)", 0, 6, 2, key="a_size")
        weapon = Weapon(models, roa, hit, dmg, surge_die, surge_plus, surge_tags, pierce_x, pierce_tag,
                        w_prec, w_crit, w_ae, concentrated_fire=w_cf)
    else:
        unit_names = [u for u, d in UNITS[a_faction].items() if d["weapons"]]
        a_unit = st.selectbox("Unit", unit_names, key=f"a_unit_{a_faction}")
        udata = UNITS[a_faction][a_unit]
        a_state = toggle_inputs(udata, f"a_{a_unit}")
        a_size = rules.unit_size(udata, a_state)
        incl_melee = st.checkbox("Include close combat weapons", value=False, key="a_melee")
        all_w = [w for w in udata["weapons"] if rules.toggles_match(w.get("requires"), a_state)]
        weapons = [w for w in all_w if incl_melee or w["phase"] == "ranged"]
        if not weapons:
            weapons = all_w
            st.info(f"{a_unit} has no ranged weapons available, showing close combat weapons.")
        w = st.selectbox("Weapon", weapons, key=f"a_w_{a_unit}_{state_key(a_state)}", format_func=weapon_label)
        a_phase = w["phase"]
        wk = f"{a_unit}_{w['name']}"
        cap = 1 if "1 model" in w["name"] else udata["models_max"]
        models = st.number_input("Models firing", 1, cap, min(udata["models_default"], cap), key=f"a_models_{wk}")
        bt = 0
        if w.get("blast"):
            bt = st.number_input("BT: units under the Blast Template", 0, 30, 1, key=f"a_bt_{wk}",
                                 help="Blast Template weapon: RoA = base RoA + BT, and the S Dice result = BT.")

        missing = w.get("roa") is None or w.get("dmg") is None
        with st.expander("Edit weapon numbers (RoA / Hit / DMG)", expanded=missing):
            if missing:
                st.warning("This weapon's RoA or DMG could not be read from the source. Enter them from your card.")
            e1, e2, e3 = st.columns(3)
            o_roa = e1.number_input("RoA (base, + BT)" if w.get("blast") else "RoA", 0, 30, 1 if w.get("roa") is None else int(w["roa"]), key=f"a_oroa_{wk}")
            o_hit = e2.selectbox("Hit", [2, 3, 4, 5, 6], index=[2, 3, 4, 5, 6].index(w["hit"]), key=f"a_ohit_{wk}",
                                 format_func=lambda v: f"{v}+")
            o_dmg = e3.number_input("DMG", 0, 20, 1 if w.get("dmg") is None else int(w["dmg"]), key=f"a_odmg_{wk}")
        weapon = Weapon(models, o_roa + bt, o_hit, o_dmg, w.get("surge", "None"), w.get("surge_plus", 0),
                        w.get("surge_tags", []), w.get("pierce_x", 0), w.get("pierce_tag", ""),
                        w.get("precision", 0), w.get("critical", 0), w.get("anti_evade", 0), bt)
        st.caption(f"**{a_unit}** - {weapon_label(w)} - range {w['rng']} - targets {w['target']}")
        # The keyword checkboxes depend on which options are ticked (an option can add keywords), so the
        # options are rendered first; the container keeps the keyword checkboxes above them on screen.
        kw_box = st.container()
        a_opts = option_inputs([o for o in udata.get("options", [])
                                if o["side"] == "attack" and rules.matches_weapon(o, w)],
                               f"a_{a_unit}", a_state, d_size, d_tags)
        kws = rules.weapon_keywords(w, a_opts)   # weapon keywords + keywords from ticked options
        weapon.concentrated_fire = rules.keyword_value(kws, "concentrated fire", min)
        with kw_box:
            if rules.find_keyword(kws, "long range"):
                long_range = st.checkbox("Long Range (target beyond normal range: -1 to hit)", key=f"a_lr_{wk}",
                                         help="This weapon has a Long Range keyword. Tick when firing at long range: "
                                              "the Hit roll is 1 harder (e.g. 3+ becomes 4+).")
            bf_kw = rules.find_keyword(kws, "burst fire")
            if bf_kw:
                burst_dice = rules.keyword_value(kws, "burst fire")
                burst_fire = st.checkbox(f"{bf_kw}: target within Burst Fire range (+{burst_dice} RoA per model)",
                                         key=f"a_bf_{wk}",
                                         help="This weapon has a Burst Fire keyword. Tick when firing within "
                                              "Burst Fire range: the RoA increases by the number in brackets.")
            if weapon.concentrated_fire:
                st.caption(f"{rules.find_keyword(kws, 'concentrated fire')}: at most {weapon.concentrated_fire} "
                           f"model(s) can be destroyed by this attack; excess damage is ignored.")

    st.markdown("**Tactical cards / other buffs**")
    atk_cards = st.multiselect("Attacker cards", cards_for("attacker", a_faction), key=f"a_cards_{a_faction}",
                               help="Shows your faction's cards plus Generic ones. Cards that don't match the weapon type (ranged/melee) or the unit's Size are ignored. Edit cards.json to add more.")
    with st.expander("Custom attacker modifiers"):
        c1, c2 = st.columns(2)
        ca_hit = c1.number_input("Hit modifier (+ = easier)", -4, 4, 0, key="ca_hit")
        ca_dpm = c2.number_input("Bonus dice per model", -3, 5, 0, key="ca_dpm")
        ca_flat = c1.number_input("Bonus dice (flat)", -20, 40, 0, key="ca_flat")
        ca_dmg = c2.number_input("Damage modifier", -3, 5, 0, key="ca_dmg")
        ca_sb = c1.number_input("Surge result bonus", 0, 5, 0, key="ca_sb")
        ca_rr = c2.selectbox("Re-roll hits", ["none", "ones", "misses"], key="ca_rr")
        ca_pr = c1.number_input("Extra PRECISION (X)", 0, 10, 0, key="ca_pr")
        ca_cr = c2.number_input("Extra CRITICAL HIT (X)", 0, 10, 0, key="ca_cr")
        ca_ae = c1.number_input("Extra ANTI-EVADE (X)", 0, 4, 0, key="ca_ae")

# ------------------------------- Run --------------------------------------- #
seed = None  # fresh random results on every run

defender = Defender(d_tags, armour, evade, evade_on, d_hp, d_shield, d_models, d_vis, d_exist)


def usable(side, names, own_size):
    used = []
    for n in names:
        c = CARDS[side][n]
        want = (c.get("when") or {}).get("own_size")
        if c.get("phase", "any") in ("any", a_phase) and (want is None or want == own_size):
            used.append(n)
    return used, [n for n in names if n not in used]


atk_used, atk_skipped = usable("attacker", atk_cards, a_size)
def_used, def_skipped = usable("defender", def_cards, d_size)
d_opts_used = [o for o in d_opts if rules.phase_ok(o, a_phase)]
d_opts_skipped = [o["name"] for o in d_opts if not rules.phase_ok(o, a_phase)]

atk_effects = ([rules.option_effects(o, d_size) for o in a_opts]
               + ([{"hit_mod": -1}] if long_range else [])
               + ([{"bonus_dice_per_model": burst_dice}] if burst_fire else [])
               + [CARDS["attacker"][c]["effects"] for c in atk_used]
               + [{"hit_mod": ca_hit, "bonus_dice_per_model": ca_dpm, "bonus_dice_flat": ca_flat, "dmg_mod": ca_dmg,
                   "surge_bonus": ca_sb, "reroll_hits": ca_rr, "precision": ca_pr, "critical": ca_cr, "anti_evade": ca_ae}])
def_effects = ([rules.option_effects(o, d_size) for o in d_opts_used]
               + [CARDS["defender"][c]["effects"] for c in def_used]
               + [{"armour_mod": cd_arm, "evade_mod": cd_eva, "grant_evade": cd_ge, "damage_reduction": cd_dr,
                   "reroll_armour": cd_rr, "shield_bonus": cd_sh, "tough": cd_to, "dodge": cd_do,
                   "total_reduction": cd_tr, "attack_pool_reduction": cd_ap, "ignore_surge": cd_is}])
atk_mods, def_mods = combine(atk_effects, AttackMods), combine(def_effects, DefendMods)

res = simulate(weapon, defender, atk_mods, def_mods, n=N_SIMS, seed=seed)
meta = res["meta"]

if atk_skipped or def_skipped or d_opts_skipped:
    st.info("Ignored (wrong attack type or unit Size for this attack): " + ", ".join(atk_skipped + def_skipped + d_opts_skipped))

st.divider()
st.subheader("Results")

dmg_arr, kills = res["damage"], res["kills"]
visible = min(d_vis, d_models)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Average damage", f"{dmg_arr.mean():.2f}")
m2.metric("Average models killed", f"{kills.mean():.2f}")
m3.metric("P(at least 1 kill)", f"{(kills >= 1).mean():.1%}")
m4.metric("P(unit wiped)", f"{(kills >= d_models).mean():.1%}")

kw = [f"{n} {meta[k_]}" for n, k_ in [("Precision", "precision"), ("Critical Hit", "critical"),
                                      ("Tough", "tough"), ("Dodge", "dodge"),
                                      ("Concentrated Fire", "concentrated_fire")] if meta[k_]]
st.caption(
    f"Effective numbers: Attack pool **{meta['pool']}** dice - Hit **{meta['hit_target']}+** - "
    f"Armour **{tn(meta['armour_target'])}** - Evade **{'off' if meta['evade_target'] is None else tn(meta['evade_target'])}** - "
    f"Surge **{meta['surge_die']}** - Damage per die **{meta['per_die_damage']}** - "
    f"First model HP incl. Shield **{meta['first_model_hp']}**" + (f" - Keywords: {', '.join(kw)}" if kw else ""))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Damage distribution**")
    vc = pd.Series(dmg_arr).value_counts(normalize=True).sort_index()
    vc = vc.reindex(range(0, int(vc.index.max()) + 1), fill_value=0)
    st.bar_chart(vc.rename("Probability"), x_label="Damage dealt", y_label="Probability")
with c2:
    st.markdown("**Models killed distribution**")
    kc = pd.Series(kills).value_counts(normalize=True).sort_index()
    kc = kc.reindex(range(0, int(visible) + 1), fill_value=0)
    st.bar_chart(kc.rename("Probability"), x_label="Models killed", y_label="Probability")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Average dice at each stage**")
    stages = pd.DataFrame({
        "Stage": ["Attack Pool", "Hits (after Precision)", "Bypass (Surge + Critical)", "Armour saved",
                  "Evaded", "Damage Pool"],
        "Avg dice": [res["attack_pool"].mean(), res["hits"].mean(), res["surged"].mean(),
                     res["armour_saved"].mean(), res["evaded"].mean(), res["damage_dice"].mean()],
    })
    st.dataframe(stages.round(2), hide_index=True, use_container_width=True)
with c2:
    st.markdown("**Kill thresholds**")
    rows = [{"Outcome": f"{k_}+ models killed", "Probability": f"{(kills >= k_).mean():.1%}"}
            for k_ in range(1, int(visible) + 1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("Damage percentiles"):
    pct = np.percentile(dmg_arr, [5, 25, 50, 75, 95])
    st.write(pd.DataFrame({"Percentile": ["5th", "25th", "median", "75th", "95th"], "Damage": pct}))
with st.expander("What is applied (unit options, cards and notes)"):
    for side, opts in (("attacker", a_opts), ("defender", d_opts_used)):
        for o in opts:
            st.markdown(f"- **{o['name']}** ({side}, {o.get('kind', 'option')}): {o.get('note', 'no note')}")
    for side, used in (("attacker", atk_used), ("defender", def_used)):
        for c in used:
            st.markdown(f"- **{c}** ({side} card): {CARDS[side][c].get('note', 'no note')}")
with st.expander("Active modifiers"):
    st.json({"attacker": atk_mods.__dict__, "defender": def_mods.__dict__})
