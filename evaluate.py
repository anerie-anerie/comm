"""
evaluate.py - scores a PII tool on benchmark.jsonl.

Recall per marker = intercepted gold entities / total gold entities.
A gold entity counts as intercepted if a predicted span with the SAME label
covers at least 50% of it. A predicted span is a false positive if it
overlaps no gold entity at all.

Run:
    python3 evaluate.py --tool scrubber            # regex + NER
    python3 evaluate.py --tool scrubber --no-ner   # regex only
    python3 evaluate.py --tool presidio            # baseline
"""
import argparse
import json
import time
from collections import defaultdict


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def load_tool(name):
    if name == "scrubber":
        import scrubber
        return scrubber.find_spans
    if name == "presidio":
        import baseline_presidio
        return baseline_presidio.find_spans
    raise SystemExit(f"unknown tool: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="scrubber", choices=["scrubber", "presidio"])
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--data", default="benchmark.jsonl")
    args = ap.parse_args()

    find_spans = load_tool(args.tool)
    rows = [json.loads(line) for line in open(args.data)]

    gold_n = defaultdict(int)
    hit_n = defaultdict(int)
    fp_n = defaultdict(int)
    pred_total = 0
    times = []

    for row in rows:
        t0 = time.perf_counter()
        preds = find_spans(row["text"], use_ner=not args.no_ner)
        times.append(time.perf_counter() - t0)
        pred_total += len(preds)

        for g in row["entities"]:
            gold_n[g["type"]] += 1
            glen = g["end"] - g["start"]
            if any(p[2] == g["type"] and
                   overlap(p[0], p[1], g["start"], g["end"]) >= 0.5 * glen
                   for p in preds):
                hit_n[g["type"]] += 1

        for p in preds:
            if not any(overlap(p[0], p[1], g["start"], g["end"]) > 0
                       for g in row["entities"]):
                fp_n[p[2]] += 1

    print(f"\nTool: {args.tool}{' (no NER)' if args.no_ner else ''}   "
          f"prompts: {len(rows)}")
    print(f"{'Marker':<10}{'Gold':>6}{'Caught':>8}{'Recall':>9}{'FalsePos':>10}")
    for t in sorted(gold_n):
        print(f"{t:<10}{gold_n[t]:>6}{hit_n[t]:>8}{hit_n[t] / gold_n[t]:>9.1%}{fp_n[t]:>10}")
    total_gold, total_hit, total_fp = sum(gold_n.values()), sum(hit_n.values()), sum(fp_n.values())
    print(f"{'OVERALL':<10}{total_gold:>6}{total_hit:>8}{total_hit / total_gold:>9.1%}{total_fp:>10}")
    precision = (pred_total - total_fp) / pred_total if pred_total else 0
    print(f"Precision (approx): {precision:.1%}   "
          f"Avg time/prompt: {1000 * sum(times) / len(times):.1f} ms")

    out = {"tool": args.tool, "no_ner": args.no_ner,
           "recall": {t: hit_n[t] / gold_n[t] for t in gold_n},
           "overall_recall": total_hit / total_gold,
           "false_positives": dict(fp_n), "precision": precision}
    fname = f"results_{args.tool}{'_noner' if args.no_ner else ''}.json"
    json.dump(out, open(fname, "w"), indent=2)
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()