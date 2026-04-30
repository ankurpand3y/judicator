#!/usr/bin/env python3
"""Judicator fixture generation orchestrator.

Pipeline (each stage fans out to as many workers as cores):
  Step 1   — extract domain seeds                       (5 parallel workers)
  Step 2   — extract position pairs (qa, code)          (2 parallel workers)
  Step 3   — derive verbosity + authority × 5 domains
             + concreteness extraction                  (11 parallel workers)
  Step 4a  — build scale_tiers (high/low) from
             SummEval + DSTC11                          (sequential)
  Step 4b  — sample self_consistency from scale_tiers   (sequential)
  Step 4c  — write yesno_probes (hand-curated)          (sequential)

All manipulators are pure-Python templates. No LLM, no API keys, no network
during generation. Re-runs are byte-identical.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import random
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulators import inject_authority, pad_response  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "judicator" / "fixtures" / "data"
SOURCES_DIR = ROOT / "data_sources"
HANDCURATED_DIR = ROOT / "scripts" / "handcurated"

DOMAINS = ["qa", "summarization", "code", "safety", "dialogue"]
PAIR_DOMAINS = ["qa", "code"]  # domains where JudgeBench provides winner/loser pairs

MT_BENCH_QUESTIONS = SOURCES_DIR / "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
MT_BENCH_REFS      = SOURCES_DIR / "FastChat/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
JB_SONNET  = SOURCES_DIR / "JudgeBench/data/dataset=judgebench,response_model=claude-3-5-sonnet-20240620.jsonl"
JB_GPT4O   = SOURCES_DIR / "JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
OFFSETBIAS = SOURCES_DIR / "offsetbias/data/evalbiasbench/biasbench.json"


# ---------- I/O helpers ----------

def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- Source loaders ----------

def _load_mt_bench() -> Tuple[Dict[int, dict], Dict[int, dict]]:
    questions = {q["question_id"]: q for q in read_jsonl(MT_BENCH_QUESTIONS)}
    refs      = {r["question_id"]: r for r in read_jsonl(MT_BENCH_REFS)}
    return questions, refs


def _load_judgebench() -> List[Dict[str, Any]]:
    items: List[dict] = []
    for path in [JB_SONNET, JB_GPT4O]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items


def _winner_loser(item: dict) -> Tuple[str, str]:
    label = item.get("label", "tie")
    if label == "A>B":
        return item["response_A"], item["response_B"]
    if label == "B>A":
        return item["response_B"], item["response_A"]
    return item["response_A"], item["response_B"]  # tie — caller should skip


def _seed(id: str, domain: str, question: str, response: str,
          source_dataset: str, license: str, **extra: Any) -> dict:
    """Flat seed: just (question, response) + provenance.

    No quality_tier here — quality labeling lives only on scale_tiers.jsonl,
    which is built directly from human-rated sources.
    """
    return {
        "id": id,
        "domain": domain,
        "question": question,
        "response": response,
        "source_dataset": source_dataset,
        "license": license,
        **extra,
    }


# ---------- Step 1: per-domain seed extractors ----------

def extract_qa_seeds(_unused: Any = None) -> List[Dict[str, Any]]:
    """MT-Bench reasoning/humanities/stem refs + JudgeBench mmlu-pro winners.

    Flat list of (question, response) pairs. No tier labels — those live
    only on scale_tiers.jsonl, built from human-rated sources.
    """
    out: List[dict] = []

    # MT-Bench reasoning/humanities/stem gpt-4 references (~30 items)
    questions, refs = _load_mt_bench()
    for qid in sorted(questions):
        q = questions[qid]
        if q["category"] not in {"reasoning", "humanities", "stem"} or qid not in refs:
            continue
        out.append(_seed(
            f"qa_{len(out)+1:03d}", "qa",
            q["turns"][0], refs[qid]["choices"][0]["turns"][0],
            "MT-Bench", "Apache 2.0",
        ))

    # JudgeBench mmlu-pro winners (deterministic sample, deduped) — fills out
    # to ~100 total. We use winners only for seed material; pair structure is
    # preserved separately in qa/position.jsonl.
    jb_items = _load_judgebench()
    mmlu = [i for i in jb_items
            if i["source"].startswith("mmlu-pro") and "math" not in i["source"]]
    seen: set = set()
    target = 100 - len(out)
    for item in mmlu:
        pid = item["pair_id"]
        if pid in seen or item.get("label") not in {"A>B", "B>A"}:
            continue
        seen.add(pid)
        winner, _ = _winner_loser(item)
        out.append(_seed(
            f"qa_{len(out)+1:03d}", "qa", item["question"], winner,
            "JudgeBench", "MIT",
        ))
        if len(out) >= 100:
            break

    return out


def extract_summarization_seeds(_unused: Any = None) -> List[Dict[str, Any]]:
    """SummEval — 100 articles × 16 summaries.

    Flat sample of 100 (article, summary) pairs across the rating distribution
    for diversity. No tier labels in this file — scale_tiers.jsonl uses
    SummEval ratings directly.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("mteb/summeval", split="test")

    triples: List[dict] = []
    for row in ds:
        article = row["text"]
        sums = row["machine_summaries"]
        rel  = row["relevance"]
        coh  = row["coherence"]
        flu  = row["fluency"]
        con  = row["consistency"]
        n = min(len(sums), len(rel), len(coh), len(flu), len(con))
        for i in range(n):
            triples.append({
                "article":    article,
                "summary":    sums[i],
                "score":      (rel[i] + coh[i] + flu[i] + con[i]) / 4.0,
                "article_id": row["id"],
            })

    # Rating-stratified sample for diversity (one summary per article
    # preferred), then fill from any remaining triples.
    triples.sort(key=lambda x: x["score"])
    seen: set = set()
    sampled: List[dict] = []
    # First pass: stride-sample one per article
    step = max(1, len(triples) // 100)
    for t in triples[::step]:
        if t["article_id"] not in seen:
            seen.add(t["article_id"])
            sampled.append(t)
            if len(sampled) >= 100:
                break
    # Fill if short
    for t in triples:
        if len(sampled) >= 100:
            break
        if t not in sampled:
            sampled.append(t)

    out: List[dict] = []
    for t in sampled[:100]:
        article_excerpt = t["article"][:1500]
        out.append(_seed(
            f"summ_{len(out)+1:03d}", "summarization",
            f"Summarize the following article:\n\n{article_excerpt}",
            t["summary"],
            "SummEval", "MIT",
        ))
    return out


def extract_code_seeds(_unused: Any = None) -> List[Dict[str, Any]]:
    """JudgeBench livecodebench winners + MT-Bench coding refs.

    Flat list of (question, response) pairs.
    """
    out: List[dict] = []

    # MT-Bench coding gpt-4 reference
    questions, refs = _load_mt_bench()
    for qid in sorted(questions):
        q = questions[qid]
        if q["category"] != "coding" or qid not in refs:
            continue
        out.append(_seed(
            f"code_{len(out)+1:03d}", "code",
            q["turns"][0], refs[qid]["choices"][0]["turns"][0],
            "MT-Bench", "Apache 2.0",
        ))

    # JudgeBench livecodebench winners (~73 items)
    jb_items = _load_judgebench()
    code_items = [i for i in jb_items if i["source"] == "livecodebench"]
    seen: set = set()
    for item in code_items:
        pid = item["pair_id"]
        if pid in seen or item.get("label") not in {"A>B", "B>A"}:
            continue
        seen.add(pid)
        winner, _ = _winner_loser(item)
        out.append(_seed(
            f"code_{len(out)+1:03d}", "code", item["question"], winner,
            "JudgeBench", "MIT",
        ))

    return out


def extract_safety_seeds(_unused: Any = None) -> List[Dict[str, Any]]:
    """BeaverTails — flat sample of safe + unsafe responses.

    No tier labels: a long safe response is not necessarily 'better' than a
    short safe one. Just (question, response) for use as seed material in
    verbosity/authority manipulators.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

    target_safe   = 60
    target_unsafe = 40
    n_safe   = 0
    n_unsafe = 0
    out: List[dict] = []
    for item in ds:
        if n_safe >= target_safe and n_unsafe >= target_unsafe:
            break
        prompt   = item["prompt"]
        response = item["response"]
        is_safe  = item["is_safe"]
        if is_safe and n_safe < target_safe:
            n_safe += 1
            out.append(_seed(
                f"safety_{len(out)+1:03d}", "safety", prompt, response,
                "BeaverTails", "CC-BY-NC-4.0", is_safe=True,
            ))
        elif (not is_safe) and n_unsafe < target_unsafe:
            n_unsafe += 1
            out.append(_seed(
                f"safety_{len(out)+1:03d}", "safety", prompt, response,
                "BeaverTails", "CC-BY-NC-4.0", is_safe=False,
            ))
    return out


def extract_dialogue_seeds(_unused: Any = None) -> List[Dict[str, Any]]:
    """DSTC11 Track 4 — open-domain dialogue.

    Flat sample across the rating distribution for diversity. Scale_tiers
    uses DSTC11 ratings directly; this file just supplies (context, response)
    seed material.
    """
    from huggingface_hub import snapshot_download  # type: ignore

    cache_root = snapshot_download(
        repo_id="mario-rc/dstc11.t4", repo_type="dataset",
    )
    zip_path = Path(cache_root) / "DSTC_11_Track_4.zip"
    z = zipfile.ZipFile(zip_path)

    items: List[dict] = []
    for ds_name in ("fed-turn", "fed-dial"):
        main_path = f"DSTC_11_Track_4/metadata/dev/en/{ds_name}/{ds_name}_eval_main.csv"
        info_path = f"DSTC_11_Track_4/metadata/dev/en/{ds_name}/{ds_name}_eval_dialoginfo.csv"

        turns_by_did: Dict[str, List[Tuple[str, str]]] = {}
        with z.open(main_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                did = row["UID"].rsplit("-", 1)[0]
                turns_by_did.setdefault(did, []).append((row["SID"], row["SEG"]))

        with z.open(info_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                if "OVERALL" not in row or not row["OVERALL"]:
                    continue
                try:
                    scores = ast.literal_eval(row["OVERALL"])
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(scores, list) or not scores:
                    continue
                quality = sum(scores) / len(scores)

                turns = turns_by_did.get(row["DID"], [])
                if len(turns) < 2:
                    continue
                response_text = turns[-1][1]
                context = "\n".join(f"{spk}: {txt}" for spk, txt in turns[:-1])
                items.append({
                    "context":  context,
                    "response": response_text,
                    "score":    quality,
                    "model":    row.get("MODEL", "unknown"),
                    "ds":       ds_name,
                })

    # Stride sample for rating diversity
    items.sort(key=lambda x: x["score"])
    target = 50
    step = max(1, len(items) // target)
    sampled = items[::step][:target]

    out: List[dict] = []
    for it in sampled:
        out.append(_seed(
            f"dialogue_{len(out)+1:03d}", "dialogue",
            it["context"], it["response"],
            "DSTC11-Track4", "Apache 2.0",
            rated_model=it["model"], sub_dataset=it["ds"],
        ))
    return out


DOMAIN_EXTRACTORS: Dict[str, Callable[[Any], List[Dict[str, Any]]]] = {
    "qa":            extract_qa_seeds,
    "summarization": extract_summarization_seeds,
    "code":          extract_code_seeds,
    "safety":        extract_safety_seeds,
    "dialogue":      extract_dialogue_seeds,
}


def step1_seeds(workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            domain: pool.submit(fn, None)
            for domain, fn in DOMAIN_EXTRACTORS.items()
        }
        for domain, fut in futures.items():
            try:
                seeds = fut.result()
                write_jsonl(DATA_DIR / domain / "seeds.jsonl", seeds)
                print(f"[step1] {domain}: {len(seeds)} seeds")
            except (NotImplementedError, FileNotFoundError) as e:
                print(f"[step1] {domain}: SKIPPED — {e}", file=sys.stderr)


# ---------- Step 2: position pairs (qa, code only) ----------

def extract_qa_pairs(_unused: Any = None) -> List[Dict[str, Any]]:
    """JudgeBench mmlu-pro winner/loser pairs for position-bias testing."""
    jb_items = _load_judgebench()
    mmlu = [i for i in jb_items
            if i["source"].startswith("mmlu-pro") and "math" not in i["source"]]
    seen: set = set()
    out: List[dict] = []
    for item in mmlu:
        pid = item["pair_id"]
        if pid in seen or item.get("label") not in {"A>B", "B>A"}:
            continue
        winner, loser = _winner_loser(item)
        if winner == loser:
            continue  # data noise — both models produced identical text
        seen.add(pid)
        out.append({
            "id":              f"qa_position_{len(out)+1:03d}",
            "bias_type":       "position",
            "domain":          "qa",
            "question":        item["question"],
            "winner_response": winner,
            "loser_response":  loser,
            "source_split":    item["source"],
            "license":         "MIT",
            "source_dataset":  "JudgeBench",
        })
        if len(out) >= 100:
            break
    return out


def extract_code_pairs(_unused: Any = None) -> List[Dict[str, Any]]:
    """JudgeBench livecodebench winner/loser pairs for position-bias testing."""
    jb_items = _load_judgebench()
    code_items = [i for i in jb_items if i["source"] == "livecodebench"]
    seen: set = set()
    out: List[dict] = []
    for item in code_items:
        pid = item["pair_id"]
        if pid in seen or item.get("label") not in {"A>B", "B>A"}:
            continue
        winner, loser = _winner_loser(item)
        if winner == loser:
            continue  # data noise — both models produced identical text
        seen.add(pid)
        out.append({
            "id":              f"code_position_{len(out)+1:03d}",
            "bias_type":       "position",
            "domain":          "code",
            "question":        item["question"],
            "winner_response": winner,
            "loser_response":  loser,
            "source_split":    "livecodebench",
            "license":         "MIT",
            "source_dataset":  "JudgeBench",
        })
    return out


PAIR_EXTRACTORS: Dict[str, Callable[[Any], List[Dict[str, Any]]]] = {
    "qa":   extract_qa_pairs,
    "code": extract_code_pairs,
}


def step2_pairs(workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            domain: pool.submit(fn, None)
            for domain, fn in PAIR_EXTRACTORS.items()
        }
        for domain, fut in futures.items():
            pairs = fut.result()
            write_jsonl(DATA_DIR / domain / "position.jsonl", pairs)
            print(f"[step2] {domain}/position: {len(pairs)} pairs")


# ---------- Step 3: manipulators ----------

def _derive_pair(seed: Dict[str, Any], domain: str, *,
                 kind: str, manipulated: str, manipulation: str) -> Dict[str, Any]:
    return {
        "id":                   f"{seed['id']}_{kind}",
        "domain":               domain,
        "bias_type":            kind,
        "question":             seed.get("question", ""),
        "original_response":    seed["response"],
        "manipulated_response": manipulated,
        "manipulation":         manipulation,
        "content_preserved":    True,
        "source_seed_id":       seed["id"],
        "license":              seed.get("license"),
        "source_dataset":       seed.get("source_dataset"),
    }


def derive_verbosity(domain: str) -> int:
    path = DATA_DIR / domain / "seeds.jsonl"
    if not path.exists():
        return 0
    seeds = read_jsonl(path)
    out = [
        _derive_pair(s, domain, kind="verbosity",
                     manipulated=pad_response(s["response"]),
                     manipulation="padding")
        for s in seeds
    ]
    write_jsonl(DATA_DIR / domain / "verbosity.jsonl", out)
    return len(out)


def derive_authority(domain: str) -> int:
    path = DATA_DIR / domain / "seeds.jsonl"
    if not path.exists():
        return 0
    seeds = read_jsonl(path)
    out = [
        _derive_pair(s, domain, kind="authority",
                     manipulated=inject_authority(s["response"], i),
                     manipulation="authority_injection")
        for i, s in enumerate(seeds)
    ]
    write_jsonl(DATA_DIR / domain / "authority.jsonl", out)
    return len(out)


def extract_concreteness(_unused: Any = None) -> int:
    """Filter OffsetBias to the 'concreteness' subcategory only.

    OffsetBias ships 6 categories testing different stylistic biases. We use
    only the 'concreteness' category — items where one response cites
    fabricated-sounding specifics (numbers, named studies, etc.) and the
    other gives an equivalently-correct but vague answer. A concreteness-
    biased judge over-rewards the fabricated-specifics version.

    response1 (preferred/correct) → vague_response
    response2 (biased/specifics-loaded) → concrete_response
    """
    data = json.load(OFFSETBIAS.open(encoding="utf-8"))
    items = data.get("concreteness", [])
    out: List[dict] = []
    for i, item in enumerate(items):
        out.append({
            "id":                f"concreteness_{i+1:03d}",
            "bias_type":         "concreteness",
            "question":          item["instruction"],
            "vague_response":    item["response1"],   # preferred / correct
            "concrete_response": item["response2"],   # specifics-loaded / biased
            "source_dataset":    "OffsetBias",
            "license":           "Apache 2.0",
        })
    write_jsonl(DATA_DIR / "universal" / "concreteness.jsonl", out)
    return len(out)


JOB_DISPATCH: Dict[str, Callable[..., int]] = {
    "verbosity":    derive_verbosity,
    "authority":    derive_authority,
    "concreteness": extract_concreteness,
}


def _run_job(kind: str, arg: Optional[str]) -> Tuple[str, Optional[str], int]:
    fn = JOB_DISPATCH[kind]
    n = fn(arg) if arg is not None else fn()
    return kind, arg, n


def step3_manipulators(workers: int) -> None:
    jobs: List[Tuple[str, Optional[str]]] = []
    for d in DOMAINS:
        jobs.append(("verbosity", d))
        jobs.append(("authority", d))
    jobs.append(("concreteness", None))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_job, kind, arg) for kind, arg in jobs]
        for fut in futs:
            kind, arg, n = fut.result()
            label = f"{kind}/{arg}" if arg else kind
            print(f"[step3] {label}: {n} items")


# ---------- Step 4a: scale_tiers (high/low from SummEval + DSTC11) ----------

def step4a_scale_tiers() -> None:
    """Build 2-bucket scale_tiers.jsonl from highest-signal sources only.

    Sources: SummEval (3-annotator means on 4 dimensions, 1-5 scale) and
    DSTC11 (5-annotator means on OVERALL scale). Both have real, multi-rater
    human ratings — the only sources where a 'high quality vs low quality'
    label is empirically grounded.

    Top quartile of ratings → tier='high', bottom quartile → tier='low'.
    Each rated triple is included (preserving article/dialogue diversity is
    handled by the underlying dataset structure, not by sampling here).
    """
    triples: List[dict] = []

    # --- SummEval ---
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("mteb/summeval", split="test")
    for row in ds:
        article = row["text"]
        sums = row["machine_summaries"]
        rel, coh, flu, con = row["relevance"], row["coherence"], row["fluency"], row["consistency"]
        n = min(len(sums), len(rel), len(coh), len(flu), len(con))
        for i in range(n):
            score = (rel[i] + coh[i] + flu[i] + con[i]) / 4.0
            triples.append({
                "domain":   "summarization",
                "question": f"Summarize the following article:\n\n{article[:1500]}",
                "response": sums[i],
                "score":    score,
                "n_annotators": 3,
                "source_dataset": "SummEval",
                "license":  "MIT",
            })

    # --- DSTC11 ---
    from huggingface_hub import snapshot_download  # type: ignore
    cache_root = snapshot_download(repo_id="mario-rc/dstc11.t4", repo_type="dataset")
    zip_path = Path(cache_root) / "DSTC_11_Track_4.zip"
    z = zipfile.ZipFile(zip_path)
    for ds_name in ("fed-turn", "fed-dial"):
        main_path = f"DSTC_11_Track_4/metadata/dev/en/{ds_name}/{ds_name}_eval_main.csv"
        info_path = f"DSTC_11_Track_4/metadata/dev/en/{ds_name}/{ds_name}_eval_dialoginfo.csv"
        turns_by_did: Dict[str, List[Tuple[str, str]]] = {}
        with z.open(main_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                did = row["UID"].rsplit("-", 1)[0]
                turns_by_did.setdefault(did, []).append((row["SID"], row["SEG"]))
        with z.open(info_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                if "OVERALL" not in row or not row["OVERALL"]:
                    continue
                try:
                    scores = ast.literal_eval(row["OVERALL"])
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(scores, list) or not scores:
                    continue
                quality = sum(scores) / len(scores)
                turns = turns_by_did.get(row["DID"], [])
                if len(turns) < 2:
                    continue
                response_text = turns[-1][1]
                context = "\n".join(f"{spk}: {txt}" for spk, txt in turns[:-1])
                triples.append({
                    "domain":   "dialogue",
                    "question": context,
                    "response": response_text,
                    "score":    quality,
                    "n_annotators": len(scores),
                    "source_dataset": "DSTC11-Track4",
                    "license":  "Apache 2.0",
                })

    # Within-source quartile cuts (SummEval and DSTC11 use different scales).
    # Cap each bucket to keep test costs proportionate with other bias tests:
    # ~100 items per bucket × 2 buckets = 200 judge calls for scale_anchoring,
    # in line with verbosity/authority (~100 calls each).
    summ_triples = [t for t in triples if t["source_dataset"] == "SummEval"]
    dstc_triples = [t for t in triples if t["source_dataset"] == "DSTC11-Track4"]

    PER_SOURCE_PER_BUCKET = 50  # 50 SummEval high + 50 DSTC11 high = 100 high

    def _split_quartile(items: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Top and bottom quartile, stride-sampled to PER_SOURCE_PER_BUCKET."""
        items.sort(key=lambda x: x["score"])
        n = len(items)
        q1 = max(1, n // 4)
        low_q  = items[:q1]
        high_q = items[-q1:]
        # Stride sample within each quartile for score diversity within bucket
        def _stride(xs: List[dict]) -> List[dict]:
            step = max(1, len(xs) // PER_SOURCE_PER_BUCKET)
            return xs[::step][:PER_SOURCE_PER_BUCKET]
        return _stride(low_q), _stride(high_q)

    summ_low, summ_high = _split_quartile(summ_triples)
    dstc_low, dstc_high = _split_quartile(dstc_triples)

    out: List[Dict[str, Any]] = []

    def _emit(triple: dict, tier: str) -> None:
        out.append({
            "id":                 f"scale_tier_{len(out)+1:03d}",
            "bias_type":          "scale_anchoring",
            "tier":               tier,
            "domain":             triple["domain"],
            "question":           triple["question"],
            "response":           triple["response"],
            "human_rating_mean":  round(triple["score"], 3),
            "n_annotators":       triple["n_annotators"],
            "source_dataset":     triple["source_dataset"],
            "license":            triple["license"],
        })

    for t in summ_high: _emit(t, "high")
    for t in dstc_high: _emit(t, "high")
    for t in summ_low:  _emit(t, "low")
    for t in dstc_low:  _emit(t, "low")

    write_jsonl(DATA_DIR / "universal" / "scale_tiers.jsonl", out)
    n_high = sum(1 for x in out if x["tier"] == "high")
    n_low  = sum(1 for x in out if x["tier"] == "low")
    print(f"[step4a] scale_tiers: {len(out)} items ({n_high} high, {n_low} low)")


# ---------- Step 4b: self_consistency (sample 40 from scale_tiers) ----------

def step4b_self_consistency() -> None:
    """Sample 40 items from scale_tiers for the self_consistency test.

    Deterministic stride sampling so re-runs produce byte-identical output.
    """
    src = DATA_DIR / "universal" / "scale_tiers.jsonl"
    items = read_jsonl(src)
    if not items:
        raise FileNotFoundError(f"{src} is empty — run step 4a first")

    target = 40
    step = max(1, len(items) // target)
    sampled = items[::step][:target]

    out: List[Dict[str, Any]] = []
    for s in sampled:
        out.append({
            "id":                    f"self_consistency_{len(out)+1:03d}",
            "bias_type":             "self_consistency",
            "domain":                s["domain"],
            "question":              s["question"],
            "response":              s["response"],
            "source_scale_tier_id":  s["id"],
            "source_dataset":        s["source_dataset"],
            "license":               s["license"],
        })
    write_jsonl(DATA_DIR / "universal" / "self_consistency.jsonl", out)
    print(f"[step4b] self_consistency: {len(out)} items")


# ---------- Step 4c: yes/no probes ----------

def step4c_yesno() -> None:
    src = HANDCURATED_DIR / "yesno_probes.json"
    if not src.exists():
        raise FileNotFoundError(
            f"Hand-author {src} first — 100 statements, 50 true / 50 false."
        )
    items = json.loads(src.read_text(encoding="utf-8"))
    for it in items:
        it.setdefault("source_dataset", "judicator-curated-v0.1")
        it.setdefault("license", "Apache-2.0")
    write_jsonl(DATA_DIR / "universal" / "yesno_probes.jsonl", items)
    print(f"[step4c] yesno_probes: {len(items)} items")


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--all",  action="store_true")
    p.add_argument("--step", action="append",
                   choices=["1", "2", "3", "4a", "4b", "4c"],
                   help="run specific step(s) only")
    args = p.parse_args()

    steps = set(args.step or [])
    if args.all:
        steps = {"1", "2", "3", "4a", "4b", "4c"}
    if not steps:
        p.error("specify --all or one or more --step")

    # Step ordering: 4b depends on 4a (reads scale_tiers.jsonl)
    if "1"  in steps: step1_seeds(args.workers)
    if "2"  in steps: step2_pairs(args.workers)
    if "3"  in steps: step3_manipulators(args.workers)
    if "4a" in steps: step4a_scale_tiers()
    if "4b" in steps: step4b_self_consistency()
    if "4c" in steps: step4c_yesno()


if __name__ == "__main__":
    main()
