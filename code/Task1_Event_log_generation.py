"""
Task 1: Event Log Generation
Process Mining & Simulation - SE4009
FAST National University

Process: Loan Application Process
Activities:
    A - Register Application
    B - Credit Check
    C - Document Verification
    D - Risk Assessment
    E - Loan Approval
    F - Loan Rejection
    G - Notify Customer
    H - Archive Case
    I - Manual Review (rare/uncommon path)

Control-flow constructs:
    Sequential:  A -> (B || C) -> D -> (E XOR F) -> G -> H
    Parallel:    B and C execute concurrently (AND)
    XOR choice:  After D, either E (Approve) or F (Reject)
    OR choice:   After Credit Check, optionally trigger I (Manual Review)
    Uncommon:    Skip directly to G (rare shortcut path)
"""

import random
import csv
import os
from itertools import product

# ─────────────────────────────────────────────
# PROCESS DEFINITION
# ─────────────────────────────────────────────

ACTIVITIES = {
    'A': 'Register Application',
    'B': 'Credit Check',
    'C': 'Document Verification',
    'D': 'Risk Assessment',
    'E': 'Loan Approval',
    'F': 'Loan Rejection',
    'G': 'Notify Customer',
    'H': 'Archive Case',
    'I': 'Manual Review',  # rare/OR branch
}

# ─────────────────────────────────────────────
# PROCESS SIMULATOR
# Generates a valid trace based on process model
# ─────────────────────────────────────────────

def generate_base_trace(uncommon_rate=0.0, num_loops=0):
    """
    Simulates one valid execution of the Loan Application Process.

    Control flow:
        1. A (sequential start)
        2. B || C (AND parallel - both must execute, order random)
        3. OR branch: optionally add I (Manual Review) after B
        4. D (sequential)
        5. XOR: E or F (exclusive choice - Approve or Reject)
        6. G (sequential)
        7. H (sequential end)

    uncommon_rate: probability of taking rare shortcut (skip D, E/F -> go straight to G)
    num_loops: number of loop-back segments to inject
    """
    trace = []

    # Step 1: Register Application (always first)
    trace.append('A')

    # Uncommon path: skip middle steps, go straight to G then H
    if random.random() < uncommon_rate:
        trace.extend(['G', 'H'])
        return trace

    # Step 2: AND parallel - B and C (randomize order)
    parallel = ['B', 'C']
    random.shuffle(parallel)
    trace.extend(parallel)

    # Step 3: OR branch - optionally add Manual Review after credit check
    if random.random() < 0.3:  # 30% chance of manual review
        trace.append('I')

    # Step 4: Risk Assessment (sequential)
    trace.append('D')

    # Step 5: XOR - Approve (70%) or Reject (30%)
    if random.random() < 0.7:
        trace.append('E')
    else:
        trace.append('F')

    # Step 6 & 7: Notify and Archive (sequential)
    trace.extend(['G', 'H'])

    # ── LOOPS ──
    # Each loop repeats a middle segment (B, C, D) once
    # Loops are inserted before D in the trace
    if num_loops > 0:
        loop_segment = ['B', 'C', 'D']
        # Find insertion point (after first D)
        d_idx = trace.index('D')
        for _ in range(num_loops):
            # Insert loop segment before the final D
            for i, act in enumerate(loop_segment):
                trace.insert(d_idx + i, act)
            d_idx += len(loop_segment)

    return trace


# ─────────────────────────────────────────────
# NOISE INJECTORS
# ─────────────────────────────────────────────

def inject_event_noise(trace):
    """Injects irrelevant noise events (not in activity set) at random positions."""
    noise_events = ['NOISE_01', 'NOISE_02', 'NOISE_03', 'NOISE_SYS', 'NOISE_ERR']
    num_noise = random.randint(1, 2)
    for _ in range(num_noise):
        pos = random.randint(0, len(trace))
        trace.insert(pos, random.choice(noise_events))
    return trace


def inject_structural_noise(trace):
    """Swaps two adjacent events to simulate ordering violations."""
    if len(trace) < 3:
        return trace
    num_swaps = random.randint(1, max(1, len(trace) // 4))
    for _ in range(num_swaps):
        idx = random.randint(0, len(trace) - 2)
        trace[idx], trace[idx + 1] = trace[idx + 1], trace[idx]
    return trace


def inject_missing_events(trace, missing_rate):
    """Randomly drops valid activities from a trace."""
    # Never drop the first and last to keep some structure
    if len(trace) <= 2:
        return trace
    middle = trace[1:-1]
    kept = [e for e in middle if random.random() > missing_rate]
    return [trace[0]] + kept + [trace[-1]]


# ─────────────────────────────────────────────
# EVENT LOG GENERATOR
# ─────────────────────────────────────────────

def generate_event_log(
    num_traces,
    noise_rate=0.0,
    missing_rate=0.0,
    uncommon_rate=0.0,
    num_loops=0,
    event_noise=False,
    structural_noise=False
):
    """
    Generates a full event log as a list of (case_id, event_id, activity) records.

    Parameters:
        num_traces     : Total number of traces to generate
        noise_rate     : Fraction of traces to apply noise to
        missing_rate   : Fraction of events to drop per trace
        uncommon_rate  : Probability of rare path at choice points
        num_loops      : Number of loop-back segments to inject per trace
        event_noise    : Whether to inject irrelevant noise events
        structural_noise: Whether to swap events (order violations)
    """
    records = []
    event_counter = 1
    num_noisy = int(num_traces * noise_rate)
    noisy_indices = set(random.sample(range(num_traces), num_noisy)) if num_noisy > 0 else set()

    for case_id in range(1, num_traces + 1):
        trace = generate_base_trace(uncommon_rate=uncommon_rate, num_loops=num_loops)

        is_noisy = case_id - 1 in noisy_indices

        # Apply missing events to ALL traces (if missing_rate > 0)
        if missing_rate > 0:
            trace = inject_missing_events(trace, missing_rate)

        # Apply noise only to selected noisy traces
        if is_noisy:
            if event_noise:
                trace = inject_event_noise(trace)
            if structural_noise:
                trace = inject_structural_noise(trace)

        for activity in trace:
            activity_name = ACTIVITIES.get(activity, activity)  # noise events won't be in dict
            records.append({
                'case_id': case_id,
                'event_id': activity,
                'activity': activity_name
            })
            event_counter += 1

    return records


# ─────────────────────────────────────────────
# SAVE TO CSV
# ─────────────────────────────────────────────

def save_log(records, filename):
    """Saves event log records to a CSV file."""
    os.makedirs('datasets', exist_ok=True)
    filepath = os.path.join('datasets', filename + '.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'event_id', 'activity'])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {filepath}  ({len(records)} events, {records[-1]['case_id']} traces)")


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(records, dataset_name):
    """Prints a summary of the generated event log."""
    from collections import Counter

    cases = {}
    for r in records:
        cases.setdefault(r['case_id'], []).append(r['event_id'])

    trace_list = [tuple(v) for v in cases.values()]
    trace_freq = Counter(trace_list)

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total traces  : {len(cases)}")
    print(f"  Total events  : {len(records)}")
    print(f"  Unique traces : {len(trace_freq)}")
    print(f"\n  Top 5 most frequent traces:")
    print(f"  {'Trace':<45} {'Freq':>5}")
    print(f"  {'-'*50}")
    for trace, freq in trace_freq.most_common(5):
        trace_str = '<' + ', '.join(trace) + '>'
        print(f"  {trace_str:<45} {freq:>5}")
    print()


# ─────────────────────────────────────────────
# MAIN — GENERATE ALL 4 DATASETS
# ─────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)  # reproducibility

    print("\n" + "="*60)
    print("   Task 1: Event Log Generation")
    print("   Process: Loan Application Process")
    print("="*60)
    print("\nActivities:")
    for k, v in ACTIVITIES.items():
        print(f"  {k} : {v}")
    print("\nProcess Flow:")
    print("  A -> (B || C) [AND] -> [OR: I?] -> D -> (E XOR F) -> G -> H")
    print("  Uncommon path: A -> G -> H (direct skip)")

    # ── Dataset 1: Clean baseline ──────────────────────────
    print("\n[1/4] Generating dataset1_clean ...")
    log1 = generate_event_log(
        num_traces=200,
        noise_rate=0.0,
        missing_rate=0.0,
        uncommon_rate=0.0,
        num_loops=0,
        event_noise=False,
        structural_noise=False
    )
    save_log(log1, 'dataset1_clean')
    print_summary(log1, 'dataset1_clean')

    # ── Dataset 2: Light noise ─────────────────────────────
    print("[2/4] Generating dataset2_light ...")
    log2 = generate_event_log(
        num_traces=200,
        noise_rate=0.07,       # 7% noisy traces
        missing_rate=0.05,     # 5% events dropped
        uncommon_rate=0.05,    # 5% chance rare path
        num_loops=1,           # 1 loop
        event_noise=True,
        structural_noise=False
    )
    save_log(log2, 'dataset2_light')
    print_summary(log2, 'dataset2_light')

    # ── Dataset 3: Medium noise ────────────────────────────
    print("[3/4] Generating dataset3_medium ...")
    log3 = generate_event_log(
        num_traces=200,
        noise_rate=0.12,       # 12% noisy traces
        missing_rate=0.10,     # 10% events dropped
        uncommon_rate=0.10,    # 10% chance rare path
        num_loops=2,           # 2 loops
        event_noise=True,
        structural_noise=True
    )
    save_log(log3, 'dataset3_medium')
    print_summary(log3, 'dataset3_medium')

    # ── Dataset 4: Heavy noise ─────────────────────────────
    print("[4/4] Generating dataset4_heavy ...")
    log4 = generate_event_log(
        num_traces=200,
        noise_rate=0.27,       # 27% noisy traces
        missing_rate=0.20,     # 20% events dropped
        uncommon_rate=0.15,    # 15% chance rare path
        num_loops=3,           # 3 loops
        event_noise=True,
        structural_noise=True
    )
    save_log(log4, 'dataset4_heavy')
    print_summary(log4, 'dataset4_heavy')

    print("="*60)
    print("  All datasets generated successfully in ./datasets/")
    print("="*60)