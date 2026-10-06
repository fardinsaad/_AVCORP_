#!/usr/bin/env python3
"""Avalon-NLU evaluation with the domain-context prompt; prior traces are the model's own outputs."""
import argparse
import datetime as _dt
import json
import os
import re
import sys
import time

RUNS_ROOT = "/share/mpsingh/fsaad/avcorp-runs"
RUNS = {  # key -> run folder and prompt variant
    "run4_1": {"folder": "base-grpo-h100-g16",           "variant": "trace"},
    "run4_2": {"folder": "base-grpo-h100-g16-r2",        "variant": "trace"},
    "run4_3": {"folder": "base-grpo-h100-g16-r3",        "variant": "trace"},
    "run5_1": {"folder": "base-reward-grpo-h100-g16",    "variant": "trace"},
    "run5_2": {"folder": "base-reward-grpo-h100-g16-r2", "variant": "trace"},
    "run5_3": {"folder": "base-reward-grpo-h100-g16-r3", "variant": "trace"},
    "run6":   {"folder": "tactic-grpo-h100-g16",         "variant": "tactic"},
    "run7":   {"folder": "tactic-reward-grpo-h100-g16",  "variant": "tactic"},
}
MAX_NEW_TOKENS = 3072  # 9_ used 2,048; its longest output was 1,482 tokens


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--stages", default="sft,grpo", help="comma list of base,sft,grpo")
    ap.add_argument("--dataset", default="avalon-nlu", choices=["avalon-nlu"])
    ap.add_argument("--data-csv", default=None, help="default: avalon_nlu.csv next to this script")
    ap.add_argument("--out-root", default=None, help=f"default: {RUNS_ROOT}/crossdomain-prompt/<dataset>")
    ap.add_argument("--dry-run", action="store_true", help="no model; completion = empty string")
    ap.add_argument("--print-prompt", default=None, metavar="WHICH",
                    help="print prompts and exit, no model: all | GAME_ID | GAME_ID:ROUND (e.g. NLU-6dBXa7:4)")
    ap.add_argument("--print-out", default=None, help="with --print-prompt: write to this file instead of stdout")
    a = ap.parse_args()
    a.stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in a.stages if s not in ("base", "sft", "grpo")]
    if bad:
        ap.error(f"unknown stage(s): {bad}")
    if a.data_csv is None:
        a.data_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avalon_nlu.csv")
    if a.out_root is None:  # separate from 9_, whose outputs would otherwise be resumed
        a.out_root = f"{RUNS_ROOT}/crossdomain-prompt/{a.dataset}"
    return a


ARGS = parse_args()
RUN = RUNS[ARGS.run]
RUN_DIR = os.path.join(RUNS_ROOT, RUN["folder"])
TACTIC = RUN["variant"] == "tactic"
PRINT_ONLY = ARGS.print_prompt is not None
NO_MODEL = ARGS.dry_run or PRINT_ONLY
OUT_DIR = os.path.join(ARGS.out_root, *(["dry-run"] if ARGS.dry_run else []), RUN["folder"])
if not PRINT_ONLY:
    os.makedirs(OUT_DIR, exist_ok=True)
_RUNLOG = os.path.join(OUT_DIR, "progress.log")


def _runlog(msg: str):
    _line = f"{_dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
    if not PRINT_ONLY:
        with open(_RUNLOG, "a") as _fh:
            _fh.write(_line + "\n")
    print(_line, flush=True)


if not NO_MODEL:  # same setup as the training notebook, without the pip install
    import site
    SHARE = "/share/mpsingh/fsaad"
    USER_BASE = os.path.join(SHARE, "python-packages")
    os.environ["PYTHONUSERBASE"] = USER_BASE
    os.environ["HF_HOME"] = os.path.join(SHARE, "hf_cache")
    os.environ["PIP_CACHE_DIR"] = os.path.join(SHARE, "pip", "cache")
    _user_site = os.path.join(
        USER_BASE, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
    )
    site.addsitedir(_user_site)
    if _user_site in sys.path:
        sys.path.remove(_user_site)
    sys.path.insert(0, _user_site)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import importlib
    for _name, _want in {"transformers": "4.56.1", "peft": "0.14.0", "accelerate": "1.4.0"}.items():
        _got = getattr(importlib.import_module(_name), "__version__", "unknown")
        if _got != _want:
            raise SystemExit(f"FATAL: {_name} {_got} loaded, training used {_want}")

import pandas as pd

df = pd.read_csv(ARGS.data_csv)


df["reasoning_gold"] = df["reasoning_gold"].fillna("")  # no gold traces outside AVCORP


def build_public_history_window(df: pd.DataFrame, window: int = 3) -> dict:  # cell 8
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


CHAIN_WINDOW = 2
public_history_lookup = build_public_history_window(df)

val_games = df['game_id'].unique()  # every row of every game; no split
val_df = df
assert len(val_games) == 20 and len(val_df) == 82, (len(val_games), len(val_df))

ALL_PLAYERS_FULL = {"P1", "P2", "P3", "P4", "P5", "P6"}  # cell 12 / tactic cell 14, edited for 6 players
N_OTHERS_WORD = "five"
GAME_SIZE = "6-player"
FLAGGED_GAMES = {"NLU-3YjxjN"}  # quest 3 logged with an all-Good team that failed (P3's re-proposal, message 54, is not logged)

# ---- domain context, from Avalon-NLU's game code and its How to Play page
GAME_RULES = (
    "Objective\n"
    "You are playing a six-player Avalon game with four Good players and two Evil players.\n"
    "Your objective is to deduce the two Evil players among the five other players using the information available to you such as round discussions, proposals, voting patterns, and quest outcomes.\n\n"
    "Game Rules\n"
    "- Six players: four Good (Merlin, Percival, and two Loyal Servants) and two Evil (Morgana and the Assassin). "
    "Roles are secret.\n"
    "- Winning: Good wins by completing three of the five quests, so Good players try to recognize one another "
    "and send teams made only of Good players. Evil wins if three quests fail. If Good completes three quests, "
    "the Assassin then names one player as Merlin, and Evil wins if the guess is right.\n"
    "- Each round has one quest. Team sizes for Quests 1 to 5 are 2, 3, 4, 3, and 4 players.\n"
    "- The leader proposes a team and may change it before calling a vote. All six players vote Yes or No. "
    "The team goes on the quest only with at least four Yes votes. Otherwise the proposal is rejected and the "
    "next player becomes leader and proposes again. If five proposals in one round are rejected, that quest fails.\n"
    "- Team members secretly play Pass or Fail. Good players always play Pass. Evil players may play Pass or Fail, "
    "and may play Pass to look trustworthy. A single Fail fails the quest. Only the outcome is announced, "
    "not who played Fail or how many Fail cards were played.\n"
    "- Players speak in turns in the chat. Any player may claim any role, and claims can be false.\n"
)
ROLE_GUIDE = (
    "Who Knows What\n"
    "- Merlin (Good) knows both Evil players but must stay hidden, because the Assassin can still win by naming "
    "Merlin. Merlin tends to steer teams and voice suspicion indirectly rather than reveal what Merlin knows.\n"
    "- Percival (Good) sees two players, Merlin and Morgana, without knowing which is which. Percival may try to "
    "protect Merlin or to test which of the two is genuine.\n"
    "- The two Loyal Servants (Good) have no special knowledge. You are one of them.\n"
    "- Morgana (Evil) knows the Assassin. Morgana appears to Percival as a possible Merlin and may pose as Merlin "
    "or as a trustworthy Good player.\n"
    "- The Assassin (Evil) knows Morgana and watches for the player who seems to know too much, to name Merlin at the end.\n"
    "- Evil players try to get onto teams, cast doubt on Good players, and appear helpful.\n"
)
EVIDENCE_GUIDE = (
    "How to Weigh the Evidence\n"
    "- A failed quest means at least one member of its team is Evil, because Good players always play Pass. "
    "If you were on a failed team, an Evil player is among the other members.\n"
    "- A passed quest is evidence that its members are Good, but not proof, since an Evil player may play Pass. "
    "A player who has been on several passed quests and on no failed quest is likely Good.\n"
    "- A player who has never been on a quest has not been tested by any quest outcome. Judge that player by the "
    "discussion and the votes.\n"
    "- Votes are weaker evidence than quest outcomes.\n"
    "- Claims of being Good or of holding a role are cheap and can be false. Weigh them against the quest record.\n"
)
RECORD_GUIDE = (
    "How to Read the Record\n"
    "- Quest Record lists, for every player, each quest that player joined and its outcome, from Quest 1 to the "
    "current quest.\n"
    "- Public History shows the current round and up to two earlier rounds. Each round gives the quest's team size "
    "and every proposal in order. \"Proposal k by Leader PX\" names the team that leader proposed; \"first A, changed "
    "before the vote to B\" means the leader proposed A and replaced it with B before calling the vote. A rejected "
    "proposal lists each player's vote and the count. After the approved proposal, the Leader, Team, and Votes lines "
    "describe the team that went on the quest, followed by the quest outcome and the score after that quest. "
    "In votes, Y means Yes and N means No.\n"
    "- Prior Round Reasoning Traces are your own traces from the previous rounds of this game.\n"
    "- Discussion Log is the chat of the current round in order. It took place before this round's quest outcome: "
    "players talk, the leader proposes, everyone votes, then the quest is played. Lines in square brackets are game "
    "events such as proposals, vote results, and quest results. Reactions to this round's outcome appear in the next "
    "round's discussion. Lines marked \"(you)\" are your own messages: you are the Investigator, not one of the "
    "players you analyze.\n"
)

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


def build_system_prompt(role_id: str) -> str: 
    others = ", ".join(_other_players(role_id))
    return (
        f"You are playing in a {GAME_SIZE} Avalon game where you investigate and deduce the two Evil players. "
        f"As such your alignment is Good and your role is the Loyal Servant (Investigator). "
        f"Your player ID is {role_id}. "
        f"You must reason about the following {N_OTHERS_WORD} players: {others}. Exactly two of them are Evil. "
        "Reason carefully from the evidence and produce only valid JSON. "
        "/no_think"
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
    'player definitively Evil, since you always submit Pass.'
)
_OUTPUT_ONLY = 'Output only the JSON object. Do not include any explanation outside the JSON.'
if TACTIC:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TACTIC + _SCHEMA_BODY + _SCHEMA_TACTICS + _SCHEMA_TAIL
else:
    SCHEMA_REMINDER = _SCHEMA_HEAD_TRACE + _SCHEMA_BODY + _SCHEMA_TAIL


def _approved_team_and_outcome(public_history: str):
    team = re.findall(r"^Team: (.+)$", public_history, re.M)
    outcome = re.findall(r"^Quest \d+ Outcome: (PASS|FAIL)$", public_history, re.M)
    assert len(team) == 1 and len(outcome) == 1, public_history
    return [p.strip() for p in team[0].split(",")], outcome[0]


QUEST_LOG = {}  # (game_id, round_id) -> (team, outcome), from each row's own public_history
for _, _r in df.iterrows():
    QUEST_LOG[(_r["game_id"], int(_r["round_id"]))] = _approved_team_and_outcome(_r["public_history"])
GAME_LAST_ROUND = df.groupby("game_id")["round_id"].max().astype(int).to_dict()


def round_state(game_id: str, round_id: int):
    """(passes, fails, game_ending) after this round's quest, from the public outcomes of quests 1..round_id."""
    outs = [QUEST_LOG[(game_id, r)][1] for r in range(1, round_id + 1)]
    passes, fails = outs.count("PASS"), outs.count("FAIL")
    return passes, fails, (passes >= 3 or fails >= 3)


for _g, _last in GAME_LAST_ROUND.items():  # every game stops at its deciding quest, and only there
    assert round_state(_g, _last)[2], _g
    assert not any(round_state(_g, _r)[2] for _r in range(1, _last)), _g


def quest_record(game_id: str, round_id: int, role_id: str) -> str:
    lines = []
    for p in sorted(ALL_PLAYERS_FULL, key=lambda x: int(x[1:])):
        joined = [(r, QUEST_LOG[(game_id, r)][1]) for r in range(1, round_id + 1) if p in QUEST_LOG[(game_id, r)][0]]
        who = f"{p} (you)" if p == role_id else p
        if joined:
            items = ", ".join(f"Q{r} {o}" for r, o in joined)
            n_pass = sum(o == "PASS" for _, o in joined)
            lines.append(f"{who}: {items} (passed {n_pass}, failed {len(joined) - n_pass})")
        else:
            lines.append(f"{who}: never on a quest")
    return "\n".join(lines)


def round_instructions(game_ending: bool, role_id: str) -> str:
    others = ", ".join(_other_players(role_id))
    scope = (f"Players to analyze: {others}. Every abduction, suspicion, level 1 belief, and deduction entry must "
             f"name one of these {N_OTHERS_WORD} players; never include yourself ({role_id}).")
    if game_ending:
        size = ("Deduction size for this round: this is the game-ending round, so your deduction must name exactly "
                "two players, the two you believe are Evil.")
    else:
        size = ("Deduction size for this round: the game continues, so your deduction must name at least one and at "
                "most three players.")
    return f"{scope}\n\n{size}"


def mark_own_lines(discussion: str, role_id: str) -> str:
    """The Investigator's own chat lines become 'PX (you): ...' (AVCORP discussions never contain the Investigator)."""
    return re.sub(rf'^{role_id}: "', f'{role_id} (you): "', discussion, flags=re.M)


def format_user_turn(row: dict, prior_chain: list = None, public_history_window: str = None) -> str:
    game_id, round_id, role_id = row["game_id"], int(row["round_id"]), row["role_id"]
    passes, fails, ending = round_state(game_id, round_id)
    situation = (
        f"Current Situation\nYou are {role_id}, a Loyal Servant (Good). This is Round {round_id} of at most 5. "
        f"After this round's quest, Good has completed {passes} and Evil has failed {fails}. "
        + (("This quest decides the game: Good has completed three quests, so the quests are won by Good and the "
            "Assassin will now try to name Merlin." if passes >= 3 else
            "This quest decides the game: Evil has failed three quests, so Evil has won.")
           + " This is the game-ending round."
           if ending else "The game continues after this round.")
    )
    chain_section = ""
    if prior_chain:
        parts = [
            f"Round {entry['round_id']} Reasoning Trace\n{entry['reasoning_gold']}"
            for entry in prior_chain
        ]
        n = len(prior_chain)
        label = "last round" if n == 1 else f"last {n} rounds"
        chain_section = (f"\nPrior Round Reasoning Traces ({label}, delimited by triple backticks)\n```\n"
                         + "\n\n".join(parts) + "\n```\n")
    history_text = public_history_window if public_history_window is not None else row['public_history'].strip()
    return (
        f"{GAME_RULES}\n{ROLE_GUIDE}\n{EVIDENCE_GUIDE}\n{RECORD_GUIDE}\n"
        f"{situation}\n\n"
        f"Quest Record (delimited by triple backticks)\n```\n{quest_record(game_id, round_id, role_id)}\n```\n\n"
        f"Public History (delimited by triple backticks)\n```\n{history_text}\n```\n"
        f"{chain_section}"
        f"\nDiscussion Log (delimited by triple backticks)\n```\n{mark_own_lines(row['discussion_log'].strip(), role_id)}\n```\n\n"
        f"{SCHEMA_REMINDER}\n\n"
        f"{round_instructions(ending, role_id)}\n\n"
        f"{_OUTPUT_ONLY}\n\n"
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


def _chain_text(completion: str) -> str:  # same fence stripping as _try_parse, no re-serialisation
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


def score_stage(stage_name: str, records: list) -> dict:  # cell 40 arithmetic
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
    clean = last[~last["game_id"].isin(FLAGGED_GAMES)]
    exact_clean = int(sum(set(str(g["predicted"]).split()) == set(str(g["ground_truth"]).split())
                          for _, g in clean.iterrows()))
    return {
        "stage": stage_name, "n": len(records), "parse_failures": parse_failures,
        "precision": micro_p, "recall": micro_r, "f1": micro_f1,
        "macro_f1": sum(per_row_f1) / len(per_row_f1) if per_row_f1 else 0.0,
        "games": n_games, "game_exact_match": exact / n_games if n_games else 0.0,
        "game_exact_count": exact, "game_hit_count": hit, "game_f1": g_f1,
        "games_excl_flagged": len(clean), "game_exact_count_excl_flagged": exact_clean,
        **{f"f1_R{rnd}": _prf(*by_round[rnd])[2] for rnd in sorted(by_round)},
        "hit_max_new_tokens": int(sum(bool(r.get("hit_max_new_tokens")) for r in records)),
    }


model = tokenizer = None
GPU_NAME = "dry-run"
if not NO_MODEL:  # cell 14 model load, plus the run's two saved adapters
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)):
        raise SystemExit("FATAL: no native bf16 on this GPU")
    COMPUTE_DTYPE = torch.bfloat16
    GPU_NAME = torch.cuda.get_device_name(0)
    MODEL_PATH = snapshot_download("Qwen/Qwen3-8B", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=COMPUTE_DTYPE, attn_implementation="sdpa",
        device_map={"": 0}, trust_remote_code=True,
    )
    GRPO_DIR = os.path.join(RUN_DIR, "grpo-qwen3-avalon")
    SFT_DIR = os.path.join(RUN_DIR, "sft-warmup")
    for _d in (GRPO_DIR, SFT_DIR):
        if not os.path.isfile(os.path.join(_d, "adapter_model.safetensors")):
            raise SystemExit(f"FATAL: no adapter_model.safetensors in {_d}")
    model = PeftModel.from_pretrained(_base, GRPO_DIR, adapter_name="grpo")
    model.load_adapter(SFT_DIR, adapter_name="sft")
    model.eval()
    _runlog(f"model {MODEL_PATH} | gpu {GPU_NAME} | sdpa bf16 | grpo={GRPO_DIR} sft={SFT_DIR}")


def run_inference(row_dict: dict, prior_chain: list = None, public_history_window: str = None,
                  max_new_tokens: int = MAX_NEW_TOKENS):  # cell 38, plus token counts
    messages = [
        {"role": "system", "content": build_system_prompt(row_dict['role_id'])},
        {"role": "user",   "content": format_user_turn(row_dict, prior_chain, public_history_window)},
    ]
    if NO_MODEL:
        return "", None, None
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return (tokenizer.decode(generated, skip_special_tokens=True),
            int(inputs['input_ids'].shape[1]), int(generated.shape[0]))


def _load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[(r["game_id"], int(r["round_id"]))] = r
    return done


def evaluate_stage(stage_name: str) -> dict:
    jsonl = os.path.join(OUT_DIR, f"pred_{stage_name}.jsonl")
    done = _load_done(jsonl)
    if done:
        _runlog(f"{stage_name}: resuming, {len(done)} rows already on disk")
    val_rows = [r.to_dict() for _, r in val_df.iterrows()]
    order = {g: i for i, g in enumerate(dict.fromkeys(r["game_id"] for r in val_rows))}  # game by game, rounds in order
    gen_rows = sorted(val_rows, key=lambda r: (order[r["game_id"]], int(r["round_id"])))
    own = {}
    n_new = 0
    t_stage = time.time()
    with open(jsonl, "a") as fh:
        for i, row in enumerate(gen_rows, 1):
            key = (row["game_id"], int(row["round_id"]))
            chain = list(own.get(row["game_id"], [])[-CHAIN_WINDOW:])
            if key in done:
                out = done[key]["output"]
            else:
                t0 = time.time()
                out, n_prompt, n_gen = run_inference(row, chain, public_history_lookup.get(key))
                rec = {
                    "game_id": row["game_id"], "round_id": int(row["round_id"]),
                    "role_id": row["role_id"], "player_roles": row["player_roles"],
                    "chain_mode": "self",
                    "chain_rounds": " ".join(str(c["round_id"]) for c in chain),
                    "prompt_tokens": n_prompt, "gen_tokens": n_gen,
                    "hit_max_new_tokens": (n_gen is not None and n_gen >= MAX_NEW_TOKENS),
                    "seconds": round(time.time() - t0, 1),
                    "output": out,
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                done[key] = rec
                n_new += 1
            own.setdefault(row["game_id"], []).append(
                {"round_id": int(row["round_id"]), "reasoning_gold": _chain_text(out)}
            )
            if i % 10 == 0:
                _runlog(f"{stage_name}: {i}/{len(gen_rows)} rows "
                        f"({n_new} generated this session, {time.time() - t_stage:.0f}s)")
    records = [dict(done[(r["game_id"], int(r["round_id"]))]) for r in val_rows]  # CSV row order
    summary = score_stage(stage_name, records)
    summary.update({"run": ARGS.run, "run_folder": RUN["folder"], "dataset": ARGS.dataset, "gpu": GPU_NAME,
                    "seconds_this_session": round(time.time() - t_stage, 1)})
    cols = ["game_id", "round_id", "role_id", "predicted", "ground_truth", "parsed", "output",
            "chain_mode", "chain_rounds", "prompt_tokens", "gen_tokens", "hit_max_new_tokens",
            "seconds"]
    pd.DataFrame(records)[cols].to_csv(os.path.join(OUT_DIR, f"eval_predictions_{stage_name}.csv"),
                                       index=False)
    return summary


def print_prompts():
    """--print-prompt: the exact system and user text for the chosen rows; no model, nothing written to OUT_DIR.
    R2+ rows show placeholders where the model's own earlier traces will go."""
    sel = ARGS.print_prompt
    rows = [r.to_dict() for _, r in val_df.iterrows()]
    if sel != "all":
        game, _, rnd = sel.partition(":")
        rows = [r for r in rows if r["game_id"] == game and (not rnd or int(r["round_id"]) == int(rnd))]
        if not rows:
            raise SystemExit(f"FATAL: no row matches {sel!r}; game ids look like NLU-6dBXa7")
    rows.sort(key=lambda r: (r["game_id"], int(r["round_id"])))
    out = []
    for r in rows:
        key = (r["game_id"], int(r["round_id"]))
        chain = [{"round_id": k, "reasoning_gold": f"<the model's own Round {k} reasoning trace>"}
                 for k in range(max(1, key[1] - CHAIN_WINDOW), key[1])]
        user = format_user_turn(r, chain, public_history_lookup.get(key))
        out.append(f"{'=' * 100}\n{key[0]} Round {key[1]} | run {ARGS.run} ({RUN['variant']} prompt) | "
                   f"game-ending: {round_state(*key)[2]} | user prompt {len(user):,} chars\n{'=' * 100}\n"
                   f"[SYSTEM]\n{build_system_prompt(r['role_id'])}\n\n[USER]\n{user}\n")
    text = "\n".join(out)
    if ARGS.print_out:
        with open(ARGS.print_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {len(rows)} prompt(s) to {ARGS.print_out}")
    else:
        print(text)


def main():
    if PRINT_ONLY:
        print_prompts()
        return
    _runlog(f"=== start | {ARGS.run} ({RUN['folder']}, {RUN['variant']}) | stages={ARGS.stages} | "
            f"dataset={ARGS.dataset} | dry_run={ARGS.dry_run} | out={OUT_DIR}")
    _runlog(f"data {ARGS.data_csv} | games {len(val_games)} rows {len(val_df)}")
    summ_path = os.path.join(OUT_DIR, "eval_summary.csv")
    results = pd.read_csv(summ_path).to_dict("records") if os.path.exists(summ_path) else []
    for stage in ARGS.stages:
        t = time.time()
        if stage == "base":
            if ARGS.dry_run:
                s = evaluate_stage(stage)
            else:
                with model.disable_adapter():
                    s = evaluate_stage(stage)
        else:
            if not ARGS.dry_run:
                model.set_adapter(stage)
            s = evaluate_stage(stage)
        results = [r for r in results if r.get("stage") != stage] + [s]
        pd.DataFrame(results).to_csv(summ_path, index=False)
        _runlog(f"{stage}: f1={s['f1']:.3f} macro_f1={s['macro_f1']:.3f} "
                f"game_end_exact={s['game_exact_count']}/{s['games']} ({s['game_exact_match']:.1%}) "
                f"game_f1={s['game_f1']:.3f} parse_fail={s['parse_failures']} "
                f"hit_max_tokens={s['hit_max_new_tokens']} | "
                f"game_end_exact_excl_flagged={s['game_exact_count_excl_flagged']}/{s['games_excl_flagged']} | "
                f"{time.time() - t:.0f}s")
    if not ARGS.dry_run:
        _runlog(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    _runlog("=== done")


if __name__ == "__main__":
    main()
