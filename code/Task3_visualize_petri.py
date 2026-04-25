"""
Task 3: Petri Net Visualization & Noise Comparison
Generates petri_dataset1_clean.png through petri_dataset4_heavy.png
Plus comparison images for datasets 2, 3, 4 vs dataset 1.
"""

import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs('images', exist_ok=True)

DATASET_NAMES = [
    ('net1.json', 'dataset1_clean',  'Dataset 1 — Clean Baseline'),
    ('net2.json', 'dataset2_light',  'Dataset 2 — Light Noise'),
    ('net3.json', 'dataset3_medium', 'Dataset 3 — Medium Noise'),
    ('net4.json', 'dataset4_heavy',  'Dataset 4 — Heavy Noise'),
]

CLEAN_PLACES = {
    'iL', 'oL', 'p_A_to_B', 'p_A_to_C', 'p_B_to_D', 'p_B_to_I',
    'p_C_to_D', 'p_C_to_I', 'p_D_to_E_F', 'p_E_F_to_G', 'p_G_to_H', 'p_I_to_D'
}

ACTIVITY_LABELS = {
    'A': 'A\nRegister', 'B': 'B\nCredit Chk', 'C': 'C\nDoc Verify',
    'D': 'D\nRisk Assess', 'E': 'E\nApproval', 'F': 'F\nRejection',
    'G': 'G\nNotify', 'H': 'H\nArchive', 'I': 'I\nManual Rev',
    'NOISE_01': 'NOISE_01', 'NOISE_02': 'NOISE_02', 'NOISE_03': 'NOISE_03',
    'NOISE_ERR': 'NOISE_ERR', 'NOISE_SYS': 'NOISE_SYS',
}


def shorten_place(p):
    """Shorten place name for display."""
    if p == 'iL': return 'iL'
    if p == 'oL': return 'oL'
    p2 = p.replace('p_', '').replace('_to_', '→')
    parts = p2.split('→')
    if len(parts) == 2:
        left = parts[0].replace('_', ',')
        right = parts[1].replace('_', ',')
        return f"{left}→{right}"
    return p2[:18]


def compute_layout(net):
    """Compute positions for all nodes using layered layout."""
    transitions = net['transitions']
    places = net['places']
    arcs = net['flow_arcs']

    # Build adjacency
    out_edges = {n: [] for n in transitions + places}
    in_edges  = {n: [] for n in transitions + places}
    for arc in arcs:
        s, d = arc['from'], arc['to']
        if s in out_edges: out_edges[s].append(d)
        if d in in_edges:  in_edges[d].append(s)

    # Assign layers via topological BFS from iL
    layer = {}
    queue = ['iL']
    layer['iL'] = 0
    visited = {'iL'}
    while queue:
        node = queue.pop(0)
        for nxt in out_edges.get(node, []):
            if nxt not in visited:
                layer[nxt] = layer[node] + 1
                visited.add(nxt)
                queue.append(nxt)
    # Assign unvisited nodes a layer based on their inputs
    for node in transitions + places:
        if node not in layer:
            preds = in_edges.get(node, [])
            if preds:
                layer[node] = max(layer.get(p, 0) for p in preds) + 1
            else:
                layer[node] = 0

    # Group by layer
    from collections import defaultdict
    layers = defaultdict(list)
    for node, l in layer.items():
        layers[l].append(node)

    max_layer = max(layers.keys()) if layers else 0
    pos = {}
    for l in range(max_layer + 1):
        nodes_in_layer = sorted(layers[l])
        n = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            x = l * 2.8
            y = (i - (n - 1) / 2) * 1.8
            pos[node] = (x, y)

    return pos


def draw_petri_net(net, title, ax, clean_places=None, highlight_spurious=False, highlight_missing=False):
    """Draw a Petri net on the given axes."""
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#1a1a2e')

    transitions = net['transitions']
    places = net['places']
    arcs = net['flow_arcs']

    pos = compute_layout(net)
    if not pos:
        ax.text(0.5, 0.5, 'No nodes', ha='center', va='center', transform=ax.transAxes)
        return

    # Determine node categories for coloring
    def get_place_color(p):
        if p == 'iL': return '#2ecc71', '#1a8a4a', 'white'   # green
        if p == 'oL': return '#e74c3c', '#a93226', 'white'   # red
        if clean_places and highlight_spurious and p not in clean_places:
            return '#ff6b6b', '#c0392b', 'white'  # spurious = red fill
        return '#dce8f5', '#2980b9', '#1a3a5c'    # normal place

    def get_transition_color(t):
        if t.startswith('NOISE'):
            return '#f39c12', '#d68910', 'white'  # noise = orange
        return '#2c3e50', '#1a252f', 'white'       # normal transition

    # Draw arcs first (behind nodes)
    drawn_pairs = set()
    for arc in arcs:
        s, d = arc['from'], arc['to']
        if s not in pos or d not in pos:
            continue
        pair = (s, d)
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        sx, sy = pos[s]
        dx, dy = pos[d]

        # Check if it's an arc between clean nodes that differs from clean model
        is_incorrect = False
        if clean_places is not None and highlight_spurious:
            # If both nodes exist in clean but this arc connects noise nodes
            s_noise = s.startswith('NOISE') or any(x in s for x in ['NOISE'])
            d_noise = d.startswith('NOISE') or any(x in d for x in ['NOISE'])
            if s_noise or d_noise:
                is_incorrect = True

        color = '#e67e22' if is_incorrect else '#7f8c8d'
        lw = 1.5 if is_incorrect else 0.9
        alpha = 0.85 if is_incorrect else 0.65

        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),arrowprops=dict(arrowstyle='->', color=color,lw=lw, alpha=alpha,connectionstyle='arc3,rad=0.0'))

    # Draw places (circles)
    place_r = 0.35
    for p in places:
        if p not in pos:
            continue
        x, y = pos[p]
        fill, edge, tc = get_place_color(p)

        # Check if missing (in clean but not in this net)
        is_missing = False
        if clean_places and highlight_missing and p in clean_places and p not in set(places):
            is_missing = True

        lw = 2.5 if (p == 'iL' or p == 'oL') else 1.5
        ls = '--' if is_missing else '-'

        circle = plt.Circle((x, y), place_r, color=fill, ec=edge, lw=lw, ls=ls, zorder=3)
        ax.add_patch(circle)

        # Label
        label = shorten_place(p)
        fontsize = 5.5 if len(label) > 10 else 6.5
        ax.text(x, y - place_r - 0.22, label, ha='center', va='top',
                fontsize=fontsize, color='#2c3e50', zorder=4,
                wrap=True)

        # Token dot for iL
        if p == 'iL':
            ax.plot(x, y, 'o', color='white', markersize=6, zorder=5)

    # Draw transitions (rectangles)
    tw, th = 0.55, 0.32
    for t in transitions:
        if t not in pos:
            continue
        x, y = pos[t]
        fill, edge, tc = get_transition_color(t)

        rect = FancyBboxPatch((x - tw/2, y - th/2), tw, th,
                               boxstyle="round,pad=0.03",
                               facecolor=fill, edgecolor=edge, lw=1.5, zorder=3)
        ax.add_patch(rect)

        label = ACTIVITY_LABELS.get(t, t)
        fontsize = 5 if t.startswith('NOISE') else 6.5
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, color=tc, fontweight='bold', zorder=4)

    # Find bounds
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    if all_x:
        margin = 1.2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.axis('off')

    # Stats box
    stats = f"Places: {len(places)}  |  Transitions: {len(transitions)}  |  Arcs: {len(net['flow_arcs'])}"
    ax.text(0.5, -0.02, stats, transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#555', style='italic')


def add_legend(ax, show_spurious=False, show_missing=False):
    handles = [
        mpatches.Patch(color='#2ecc71', label='Source place (iL)'),
        mpatches.Patch(color='#e74c3c', label='Sink place (oL)'),
        mpatches.Patch(color='#dce8f5', ec='#2980b9', label='Normal place'),
        mpatches.Patch(color='#2c3e50', label='Transition'),
    ]
    if show_spurious:
        handles.append(mpatches.Patch(color='#ff6b6b', label='Spurious place'))
    if show_missing:
        handles.append(mpatches.Patch(facecolor='white', ec='#2980b9',
                                       linestyle='--', label='Missing place'))
    handles.append(mpatches.Patch(color='#f39c12', label='Noise transition'))
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.9, edgecolor='#ccc')


# ── STEP 1: Individual Petri net images ─────────────────────────────────────
print("Generating individual Petri net images...")
nets = []
for json_file, ds_name, ds_title in DATASET_NAMES:
    with open(f'datasets/{json_file}') as f:
        net = json.load(f)
    nets.append(net)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')

    draw_petri_net(net, ds_title, ax)
    add_legend(ax)

    # Big title
    fig.suptitle(f'Petri Net α(L) — {ds_title}', fontsize=14, fontweight='bold',
                 y=0.98, color='#1a1a2e')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = f'images/petri_{ds_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ── STEP 2: Comparison images (noisy vs clean) ───────────────────────────────
print("\nGenerating comparison images...")
clean_net = nets[0]
clean_place_set = set(clean_net['places'])

noisy_datasets = [
    (nets[1], 'dataset2_light',  'Dataset 2 — Light Noise'),
    (nets[2], 'dataset3_medium', 'Dataset 3 — Medium Noise'),
    (nets[3], 'dataset4_heavy',  'Dataset 4 — Heavy Noise'),
]

for noisy_net, ds_name, ds_title in noisy_datasets:
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('white')

    # Left: clean
    draw_petri_net(clean_net, 'Dataset 1 — Clean Baseline', axes[0])

    # Right: noisy with annotations
    draw_petri_net(noisy_net, ds_title, axes[1],
                   clean_places=clean_place_set,
                   highlight_spurious=True)

    add_legend(axes[1], show_spurious=True)

    # Compute stats for annotation
    noisy_places = set(noisy_net['places'])
    spurious = noisy_places - clean_place_set
    missing  = clean_place_set - noisy_places

    # Annotation box
    ann = (f"Spurious places (red): {len(spurious)}\n"
           f"Missing places: {len(missing)}\n"
           f"Noise transitions: {sum(1 for t in noisy_net['transitions'] if t.startswith('NOISE'))}")
    axes[1].text(0.01, 0.99, ann, transform=axes[1].transAxes,
                 va='top', ha='left', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9, edgecolor='#f39c12'))

    fig.suptitle(f'Side-by-Side Comparison: Clean vs {ds_title}',
                 fontsize=14, fontweight='bold', y=0.99, color='#1a1a2e')

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    out_path = f'images/comparison_{ds_name}_vs_clean.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ── STEP 3: Summary table image ───────────────────────────────────────────────
print("\nGenerating summary table image...")
clean_place_set = set(nets[0]['places'])

rows = []
ds_labels = ['Dataset 1\n(Clean)', 'Dataset 2\n(Light)', 'Dataset 3\n(Medium)', 'Dataset 4\n(Heavy)']
for i, net in enumerate(nets):
    np_set = set(net['places'])
    spurious = len(np_set - clean_place_set) if i > 0 else 0
    missing  = len(clean_place_set - np_set) if i > 0 else 0
    rows.append([
        len(net['places']),
        len(net['transitions']),
        len(net['flow_arcs']),
        spurious,
        missing,
    ])

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('white')
ax.axis('off')

col_labels = ['Places\nDiscovered', 'Transitions\nDiscovered', 'Arcs\nDiscovered',
              'Spurious\nPlaces', 'Missing\nPlaces']
table_data = [row for row in rows]

tbl = ax.table(
    cellText=table_data,
    rowLabels=ds_labels,
    colLabels=col_labels,
    cellLoc='center',
    rowLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.4, 2.2)

# Style header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

# Style row labels
for i in range(1, 5):
    tbl[(i, -1)].set_facecolor('#dce8f5')
    tbl[(i, -1)].set_text_props(fontweight='bold', color='#1a3a5c')

# Highlight spurious/missing columns in noisy rows
for i in range(2, 5):  # datasets 2,3,4
    for j in [3, 4]:   # spurious, missing columns
        val = table_data[i-1][j]
        if val > 0:
            tbl[(i, j)].set_facecolor('#ffe0e0')
            tbl[(i, j)].set_text_props(color='#c0392b', fontweight='bold')

# Dataset 1 row
for j in range(5):
    tbl[(1, 0)].set_facecolor('#e8f5e9')

ax.set_title('Task 3.3 — Comparison Summary Table', fontsize=13,
             fontweight='bold', pad=20, color='#1a1a2e')

plt.tight_layout()
out_path = 'images/comparison_summary_table.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out_path}")

print("\nAll images generated successfully!")
print("\nSummary statistics:")
print(f"{'Dataset':<20} {'Places':>8} {'Trans':>8} {'Arcs':>8} {'Spurious':>10} {'Missing':>10}")
print("-" * 66)
for i, (net, row) in enumerate(zip(nets, rows)):
    label = ['Clean','Light','Medium','Heavy'][i]
    print(f"{'Dataset '+str(i+1)+' ('+label+')':<20} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>10} {row[4]:>10}")
    