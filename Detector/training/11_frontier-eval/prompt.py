"""
AVCORP validation data, prompt and scoring, copied from
training/8_eval-test-with-model-trace/8_eval_model_trace.py with the tactic
variant on. The system prompt drops "/no_think".

    python prompt.py            split summary
    python prompt.py G015:3     print the round-3 prompt of game G015
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "Deception-Dataset.csv"
TACTIC = True
CHAIN_WINDOW = 2

df = pd.read_csv(DATA_CSV)


def build_gold_with_tactics(row: dict) -> str:
    trace = json.loads(row['reasoning_gold'])
    tactic_info = json.loads(row['matrix_tactic_scale'])
    trace['tactics'] = [
        {"player": p, "tactic": info.get("tactic")}
        for p, info in tactic_info.items()
    ]
    return json.dumps(trace)


GOLD_COL = "reasoning_gold_with_tactics" if TACTIC else "reasoning_gold"
if TACTIC:
    df['reasoning_gold_with_tactics'] = df.apply(
        lambda r: build_gold_with_tactics(r.to_dict()), axis=1
    )


def build_public_history_window(df: pd.DataFrame, window: int = 3) -> dict:
    lookup = {}
    for game_id, group in df.groupby('game_id'):
        sorted_group = group.sort_values('round_id')
        rows = list(sorted_group.iterrows())
        for i, (_, row) in enumerate(rows):
            rid = int(row['round_id'])
            start = max(0, i - (window - 1))
            parts = [f"[R{int(r['round_id'])}] {r['public_history'].strip()}" for _, r in rows[start:i + 1]]
            lookup[(game_id, rid)] = "\n".join(parts)
    return lookup


public_history_lookup = build_public_history_window(df)

game_end_round = df.groupby('game_id')['round_id'].max()
train_games, val_games = train_test_split(
    game_end_round.index.to_numpy(), test_size=0.10, random_state=42, stratify=game_end_round.values
)
val_df = df[df['game_id'].isin(val_games)]
assert len(val_games) == 25 and len(val_df) == 100, (len(val_games), len(val_df))

ALL_PLAYERS_FULL = {"P1", "P2", "P3", "P4", "P5"}
N_OTHERS_WORD = "four"
GAME_SIZE = "5-player"
RULES = ""

TACTIC_VOCAB = [
    "Bayesian hedging", "Bold falsehoods", "Bullshitting", "Cherry-picking",
    "Confusion tactics", "Defensive disclosure", "Deflection", "Evidence sharing",
    "Face-saving blather", "False consensus", "Gaslighting", "Half-truths",
    "Hard lying", "Honest mistake", "Misframing", "Misleading emphasis",
    "Noisy sincerity", "Obfuscatory nonsense", "Omission to save face",
    "Over-explaining", "Overstated contribution", "Perspective-taking",
    "Pragmatic silence", "Rational justification", "Rationalized error",
    "Reverse accusations", "Self-deception", "Self-justification",
    "Sincere false belief", "Strategic fabrication", "Strategic omission",
    "Strategic uncertainty", "Technobabble", "Truthful misdirection",
    "Vague filler", "Weaponized candor", "Withholding info (WIFOM)",
]


def _other_players(role_id):
    return sorted(ALL_PLAYERS_FULL - {role_id})


def build_system_prompt(role_id: str, no_think: bool = False) -> str:
    """no_think=True gives Qwen's system prompt exactly, "/no_think" included."""
    others = ", ".join(_other_players(role_id))
    return (
        f"You are the Investigator in a {GAME_SIZE} Avalon game. "
        f"Your player ID is {role_id} and your alignment is Good. "
        f"You must reason about the following {N_OTHERS_WORD} players: {others}. "
        f"{RULES}"
        "Reason carefully from the evidence and produce only valid JSON."
        + (" /no_think" if no_think else "")
    )


_SCHEMA_HEAD_TRACE = (
    "Given the public history and discussion log, produce a structured JSON reasoning trace "
    "with exactly the following six keys.\n\n"
)
_SCHEMA_HEAD_TACTIC = (
    "Given the public history and discussion log, produce a structured JSON reasoning trace "
    "with exactly the following seven keys.\n\n"
)
_SCHEMA_BODY = (
    f'abduction: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player, good_expl, evil_expl, and choice. '
    'The choice field must be either Good or Evil.\n\n'
    f'suspicion: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player, level, and reason. '
    'Valid levels are Minimal, Mild, Moderate, Strong, and Extreme.\n\n'
    'depth: an integer, either 1 or 2. '
    'Use 2 only when at least one player has Strong or Extreme suspicion.\n\n'
    'beliefs: a list of belief objects. '
    f'Level 1 beliefs must exist for all {N_OTHERS_WORD} other players. '
    'Level 2 beliefs may exist only for players with Strong or Extreme suspicion. '
    'Each belief object has the fields level, player, and content.\n\n'
    'statement: a string containing your public statement for this round.\n\n'
    'deduction: a list of player IDs you believe are Evil, for example ["P3", "P4"].\n\n'
)
_SCHEMA_TACTICS = (
    f'tactics: a list of {N_OTHERS_WORD} objects, one per other player. '
    'Each object has the fields player and tactic. '
    f'The tactic field must be exactly one of: {", ".join(TACTIC_VOCAB)}.\n\n'
)
_SCHEMA_TAIL = (
    'Self-Knowledge Elimination: you know your own alignment and actions with certainty. '
    'If a quest\'s outcome contradicts your known-Good behavior, eliminate suspects using the '
    'team\'s known composition, confidence scaling with how few possibilities remain. One '
    'remaining suspect means a definitive conclusion (Extreme suspicion, Evil abduction choice), '
    'not just strong suspicion. Example: a failed 2-player quest you were on makes the other '
    'player definitively Evil, since you always submit Pass.\n\n'
    'Output only the JSON object. Do not include any explanation outside the JSON.'
)
if TACTIC:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TACTIC + _SCHEMA_BODY + _SCHEMA_TACTICS + _SCHEMA_TAIL
else:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TRACE + _SCHEMA_BODY + _SCHEMA_TAIL


def format_user_turn(row: dict, prior_chain: list = None, public_history_window: str = None) -> str:
    chain_section = ""
    if prior_chain:
        parts = [
            f"Round {entry['round_id']} Reasoning Trace\n{entry['reasoning_gold']}"
            for entry in prior_chain
        ]
        n = len(prior_chain)
        label = "last round" if n == 1 else f"last {n} rounds"
        chain_section = f"\nPrior Round Reasoning Traces ({label})\n" + "\n\n".join(parts) + "\n"
    history_text = public_history_window if public_history_window is not None else row['public_history'].strip()
    return (
        f"Public History\n{history_text}\n"
        f"{chain_section}"
        f"\nDiscussion Log\n{row['discussion_log'].strip()}\n\n"
        f"{SCHEMA_REMINDER}\n\n"
        f"Produce the JSON reasoning trace:"
    )


def _try_parse(completion):
    if isinstance(completion, list):
        completion = completion[-1].get("content", "") if completion and isinstance(completion[-1], dict) else ""
    elif isinstance(completion, dict):
        completion = completion.get("content", "")
    if not isinstance(completion, str):
        return None
    completion = re.sub(r"```(?:json)?\s*", "", completion).strip().rstrip("`")
    try:
        return json.loads(completion)
    except Exception:
        pass
    match = re.search(r"(\{.*\})", completion, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def _chain_text(completion: str) -> str:
    """The trace fed to later rounds: code fences stripped, text otherwise untouched."""
    return re.sub(r"```(?:json)?\s*", "", completion).strip().rstrip("`")


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def _deduction_from_output(text: str, role_id: str):
    parsed = _try_parse(text)
    others = set(_other_players(role_id))
    if not isinstance(parsed, dict):
        return set(), False
    raw = parsed.get("deduction")
    if not isinstance(raw, list):
        return set(), True
    return {p for p in raw if p in others}, True


def score_stage(stage_name: str, records: list) -> dict:
    tp = fp = fn = 0
    per_row_f1, parse_failures, by_round = [], 0, {}
    for rec in records:
        pred, ok = _deduction_from_output(rec["output"], rec["role_id"])
        if not ok:
            parse_failures += 1
        others = set(_other_players(rec["role_id"]))
        gt = {p for p, v in json.loads(rec["player_roles"]).items() if v == "Evil" and p in others}
        t, f_pos, f_neg = len(pred & gt), len(pred - gt), len(gt - pred)
        tp += t; fp += f_pos; fn += f_neg
        acc = by_round.setdefault(int(rec["round_id"]), [0, 0, 0])
        acc[0] += t; acc[1] += f_pos; acc[2] += f_neg
        prec = t / len(pred) if pred else 0.0
        rec_ = t / len(gt) if gt else 0.0
        per_row_f1.append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
        rec["predicted"] = " ".join(sorted(pred))
        rec["ground_truth"] = " ".join(sorted(gt))
        rec["parsed"] = ok
    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)
    rec_df = pd.DataFrame(records)
    last = rec_df.loc[rec_df.groupby("game_id")["round_id"].idxmax()]
    g_tp = g_fp = g_fn = 0
    exact = hit = 0
    for _, g in last.iterrows():
        pred = set(g["predicted"].split()) if g["predicted"] else set()
        gt = set(g["ground_truth"].split()) if g["ground_truth"] else set()
        g_tp += len(pred & gt); g_fp += len(pred - gt); g_fn += len(gt - pred)
        exact += pred == gt
        hit += bool(pred & gt)
    n_games = len(last)
    g_p, g_r, g_f1 = _prf(g_tp, g_fp, g_fn)
    return {
        "stage": stage_name, "n": len(records), "parse_failures": parse_failures,
        "precision": micro_p, "recall": micro_r, "f1": micro_f1,
        "macro_f1": sum(per_row_f1) / len(per_row_f1) if per_row_f1 else 0.0,
        "games": n_games, "game_exact_match": exact / n_games if n_games else 0.0,
        "game_exact_count": exact, "game_hit_count": hit, "game_f1": g_f1,
        **{f"f1_R{rnd}": _prf(*by_round[rnd])[2] for rnd in sorted(by_round)},
        "hit_max_new_tokens": int(sum(bool(r.get("hit_max_new_tokens")) for r in records)),
    }


VAL_ROWS = [r.to_dict() for _, r in val_df.iterrows()]
_ORDER = {g: i for i, g in enumerate(dict.fromkeys(r["game_id"] for r in VAL_ROWS))}
GAMES = list(_ORDER)
ROWS_BY_GAME = {g: sorted((r for r in VAL_ROWS if r["game_id"] == g), key=lambda r: int(r["round_id"]))
                for g in GAMES}


# Claude Opus 5.5 refuses "reasoning trace" wording as reasoning extraction, so its neutral run asks for the same JSON in plainer words.
NEUTRAL = [
    ("You must reason about the following", "You must assess the following"),
    ("Reason carefully from the evidence", "Weigh the evidence carefully"),
    ("Prior Round Reasoning Traces", "Earlier Round Reports"),
    (" Reasoning Trace\n", " Report\n"),
    ("produce a structured JSON reasoning trace", "produce a structured JSON report"),
    ("fields player, level, and reason.", "fields player, level, and basis."),
    ("Produce the JSON reasoning trace:", "Produce the JSON report:"),
]


def messages_for(row: dict, prior_chain: list, no_think: bool = False, neutral: bool = False):
    key = (row["game_id"], int(row["round_id"]))
    system = build_system_prompt(row["role_id"], no_think=no_think)
    user = format_user_turn(row, prior_chain, public_history_lookup.get(key))
    for old, new in (NEUTRAL if neutral else []):
        system, user = system.replace(old, new), user.replace(old, new)
    return system, user


if __name__ == "__main__":
    if len(sys.argv) > 1:
        g, r = sys.argv[1].split(":")
        row = next(x for x in ROWS_BY_GAME[g] if int(x["round_id"]) == int(r))
        s, u = messages_for(row, [])
        print("=== SYSTEM ===\n" + s + "\n\n=== USER (prior traces empty here; the runner fills them) ===\n" + u)
    else:
        print(f"{len(GAMES)} games, {len(VAL_ROWS)} rows; rounds per game "
              f"{pd.Series([len(v) for v in ROWS_BY_GAME.values()]).value_counts().sort_index().to_dict()}")
