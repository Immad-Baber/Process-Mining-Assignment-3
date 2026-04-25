"""
Task 2: Alpha Algorithm Implementation (from scratch)
Process Mining & Simulation - SE4009

Implements:
1) Unique traces and frequencies
2) Event sets TL, TI, TO
3) Footprint matrix
4) Relation sets XL, YL, PL, FL
5) Petri net alpha(L)
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from itertools import combinations


def parse_event_log(csv_path):
    """Read CSV and return traces as ordered tuples of event IDs."""
    cases = defaultdict(list)

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        required = {'case_id', 'event_id'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "Input CSV must contain headers: case_id,event_id (activity optional)."
            )
        for row in reader:
            case_id = str(row['case_id']).strip()
            event_id = str(row['event_id']).strip()
            if not case_id or not event_id:
                continue
            cases[case_id].append(event_id)

    # Keep case ordering stable by numeric ID when possible.
    def case_sort_key(case):
        return (0, int(case)) if case.isdigit() else (1, case)

    ordered_case_ids = sorted(cases.keys(), key=case_sort_key)
    traces = [tuple(cases[cid]) for cid in ordered_case_ids if cases[cid]]
    return traces


def unique_trace_frequencies(traces):
    return Counter(traces)


def compute_event_sets(traces):
    tl = sorted({act for trace in traces for act in trace})
    ti = sorted({trace[0] for trace in traces if trace})
    to = sorted({trace[-1] for trace in traces if trace})
    return tl, ti, to


def compute_direct_succession(traces):
    ds = set()
    loops_1 = set()
    for trace in traces:
        for i in range(len(trace) - 1):
            a, b = trace[i], trace[i + 1]
            ds.add((a, b))
            if a == b:
                loops_1.add(a)
    return ds, loops_1


def relation_symbol(a, b, direct_succession):
    ab = (a, b) in direct_succession
    ba = (b, a) in direct_succession
    if ab and not ba:
        return "->"
    if ba and not ab:
        return "<-"
    if ab and ba:
        return "||"
    return "#"


def compute_footprint_matrix(tl, direct_succession):
    matrix = {}
    for a in tl:
        for b in tl:
            matrix[(a, b)] = relation_symbol(a, b, direct_succession)
    return matrix


def powerset_nonempty(items):
    items = list(items)
    subsets = []
    for size in range(1, len(items) + 1):
        for comb in combinations(items, size):
            subsets.append(frozenset(comb))
    return subsets


def is_all_sharp(items, matrix):
    items = list(items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if matrix[(items[i], items[j])] != "#":
                return False
    return True


def is_all_causal(left_set, right_set, matrix):
    for a in left_set:
        for b in right_set:
            if matrix[(a, b)] != "->":
                return False
    return True


def compute_xl(tl, matrix):
    """XL contains all valid pairs (A, B) per alpha algorithm."""
    all_subsets = powerset_nonempty(tl)
    xl = []
    for left in all_subsets:
        if not is_all_sharp(left, matrix):
            continue
        for right in all_subsets:
            if not is_all_sharp(right, matrix):
                continue
            if is_all_causal(left, right, matrix):
                xl.append((left, right))
    return xl


def compute_yl(xl):
    """Keep maximal pairs only (remove subsumed pairs)."""
    yl = []
    for i, (a1, b1) in enumerate(xl):
        subsumed = False
        for j, (a2, b2) in enumerate(xl):
            if i == j:
                continue
            if a1.issubset(a2) and b1.issubset(b2) and (a1 != a2 or b1 != b2):
                subsumed = True
                break
        if not subsumed:
            yl.append((a1, b1))
    return yl


def format_set_list(values):
    return "{" + ", ".join(values) + "}"


def pair_to_place_name(a_set, b_set):
    left = "_".join(sorted(a_set))
    right = "_".join(sorted(b_set))
    return f"p_{left}_to_{right}"


def compute_pl_fl(yl, ti, to):
    places = ["iL", "oL"]
    flow = set()

    # Source and sink arcs
    for t in ti:
        flow.add(("iL", t))
    for t in to:
        flow.add((t, "oL"))

    # Internal places from YL pairs
    for left, right in yl:
        p = pair_to_place_name(left, right)
        places.append(p)
        for a in left:
            flow.add((a, p))
        for b in right:
            flow.add((p, b))

    return sorted(places), sorted(flow)


def build_alpha_petri_net(tl, ti, to, yl, pl, fl):
    transitions = sorted(tl)
    places = sorted(pl)
    flow = sorted(fl)
    return {
        "transitions": transitions,
        "places": places,
        "source_place": "iL",
        "sink_place": "oL",
        "start_transitions": sorted(ti),
        "end_transitions": sorted(to),
        "yl_pairs": [
            {
                "A": sorted(list(a)),
                "B": sorted(list(b)),
                "place": pair_to_place_name(a, b),
            }
            for a, b in yl
        ],
        "flow_arcs": [{"from": src, "to": dst} for src, dst in flow],
    }


def print_step_1(trace_freq):
    print("\n=== STEP 1: Event Log L ===")
    print(f"{'Trace':<60} {'Frequency':>10}")
    print("-" * 75)
    for trace, freq in sorted(trace_freq.items(), key=lambda x: (-x[1], x[0])):
        trace_str = "<" + ", ".join(trace) + ">"
        print(f"{trace_str:<60} {freq:>10}")


def print_step_2(tl, ti, to):
    print("\n=== STEP 2: Event Sets - TL, TI, TO ===")
    print(f"TL (all activities)   = {format_set_list(tl)}")
    print(f"TI (start activities) = {format_set_list(ti)}")
    print(f"TO (end activities)   = {format_set_list(to)}")


def print_step_3(tl, matrix, short_loops):
    print("\n=== STEP 3: Footprint Matrix ===")
    print("Symbols: -> causal, <- reverse causal, || parallel, # unrelated")

    col_width = 5
    header = " " * 5 + "".join(f"{a:>{col_width}}" for a in tl)
    print(header)
    print("-" * len(header))

    for a in tl:
        row = f"{a:>4} "
        for b in tl:
            row += f"{matrix[(a, b)]:>{col_width}}"
        print(row)

    if short_loops:
        print("\nWARNING: Short loops detected (a -> a) for:")
        print("  " + ", ".join(sorted(short_loops)))
    else:
        print("\nNo short loops (a -> a) detected.")


def print_step_4(xl, yl, pl, fl):
    print("\n=== STEP 4: Relation Sets (XL, YL, PL, FL) ===")

    print("\nXL (candidate pairs):")
    for left, right in yl_sort(xl):
        print(f"  ({format_set_list(sorted(left))}, {format_set_list(sorted(right))})")

    print("\nYL (maximal pairs):")
    for left, right in yl_sort(yl):
        print(f"  ({format_set_list(sorted(left))}, {format_set_list(sorted(right))})")

    print("\nPL (places):")
    print("  " + ", ".join(pl))

    print("\nFL (flow arcs):")
    for src, dst in fl:
        print(f"  {src} -> {dst}")


def print_step_5(net):
    print("\n=== STEP 5: Build Petri Net alpha(L) ===")
    print(f"Transitions (T): {format_set_list(net['transitions'])}")
    print(f"Places (P):      {format_set_list(net['places'])}")
    print(f"Source place:    {net['source_place']}")
    print(f"Sink place:      {net['sink_place']}")
    print(f"Arcs count:      {len(net['flow_arcs'])}")


def yl_sort(pairs):
    return sorted(
        pairs,
        key=lambda pair: (tuple(sorted(pair[0])), tuple(sorted(pair[1]))),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpha Algorithm on an event log CSV."
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "datasets",
            "dataset1_clean.csv",
        ),
        help="Path to input event log CSV.",
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional path to export discovered Petri net JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traces = parse_event_log(args.dataset)
    if not traces:
        raise ValueError("No traces found in input CSV.")

    trace_freq = unique_trace_frequencies(traces)
    tl, ti, to = compute_event_sets(traces)
    direct_succession, short_loops = compute_direct_succession(traces)
    matrix = compute_footprint_matrix(tl, direct_succession)
    xl = compute_xl(tl, matrix)
    yl = compute_yl(xl)
    pl, fl = compute_pl_fl(yl, ti, to)
    net = build_alpha_petri_net(tl, ti, to, yl, pl, fl)

    print("\n" + "=" * 78)
    print("Task 2: Alpha Algorithm Implementation")
    print(f"Input dataset: {args.dataset}")
    print("=" * 78)

    print_step_1(trace_freq)
    print_step_2(tl, ti, to)
    print_step_3(tl, matrix, short_loops)
    print_step_4(xl, yl, pl, fl)
    print_step_5(net)

    if args.export_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, "w") as f:
            json.dump(net, f, indent=2)
        print(f"\nExported Petri net JSON: {args.export_json}")


if __name__ == "__main__":
    main()
