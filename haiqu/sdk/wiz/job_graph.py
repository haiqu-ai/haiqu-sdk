"""
Job execution graph visualizations.
"""

try:
    from IPython.display import display, HTML
except ImportError:
    print("Jupyter required! Called function intended to work in Notebook/Lab environment.")

NODE_DESCRIPTIONS: dict[str, str] = {
    "Packing": "Packs N circuits side-by-side into one job to fill device qubits; unpacks per-circuit results afterward.",
    "Transpile": (
        "Maps logical qubits to physical qubits and decomposes gates to the"
        " device's native gate set via routing and optimization passes."
    ),
    "Observable Splitter": (
        "Splits a multi-observable task into one task per Pauli observable" " so downstream layers process each independently."
    ),
    "Advanced Observable Mitigation": (
        "Rescales expectation values using auxiliary reference circuits to correct"
        " for systematic gate errors; active in advanced mode only."
    ),
    "Advanced Distribution Mitigation": (
        "Applies structured noise correction to bitstring distributions using"
        " auxiliary calibration circuits; active in advanced mode only."
    ),
    "Noise Tailoring": "Applies noise tailoring to reshape device noise into a more structured form.",
    "Dynamical Decoupling": (
        "Inserts X/Y/Z pulse sequences into idle qubit windows at pulse level" " to cancel dephasing and depolarization."
    ),
    "Observable Readout Correction": (
        "Applies readout error correction to observable measurement results" " when advanced mitigation is not enabled."
    ),
    "QWC Compute": (
        "Partitions Pauli strings into qubit-wise commuting (QWC) families,"
        " appends joint measurement circuits, and computes expectation values"
        " from shot counts."
    ),
    "Advanced Readout Mitigation": "Internal Haiqu advanced readout mitigation layer.",
    "Readout Mitigation": (
        "Applies calibration-matrix readout error correction to bitstring distributions"
        " using auxiliary calibration circuits merged back into the main stream."
    ),
    "Merge": "Merges the main circuit stream with auxiliary calibration task streams into a single device batch.",
}


def build_run_job_graph(
    device_id: str,
    uses_observables: bool,
    uses_mitigation: bool,
    uses_packing: bool,
    uses_transpilation: bool = True,
    use_advanced: bool = True,
    use_noise_tailoring: bool = False,
    use_dd: bool = True,
    use_readout: bool = True,
):
    """Build the execution flow MultiDiGraph for a run job.

    Node names mirror the actual layer classes in run
    (some layer names are abstracted for confidentiality):
      Packing, Transpile, Observable Splitter, Advanced Observable Mitigation,
      Advanced Distribution Mitigation, Noise Tailoring, Dynamical Decoupling,
      Observable Readout Correction, QWC Compute, Advanced Readout Mitigation,
      Readout Mitigation, Merge, <device_id>.

    Graph attributes set on the returned graph:
      G.graph['device']     – the device node label
      G.graph['arch_edges'] – set of (u, v) skip edges representing aux-task branches
                               (Advanced Observable Mitigation, Advanced Distribution Mitigation,
                                or Readout Mitigation → Merge; at most one per path)
    """
    import networkx as nx

    device = device_id or "Device"
    G = nx.MultiDiGraph()
    prev = "Input"
    arch_edges: set = set()

    def chain(name):
        nonlocal prev
        G.add_edge(prev, name)
        prev = name

    if uses_packing:
        chain("Packing")

    if uses_transpilation:
        chain("Transpile")

    if uses_observables:
        chain("Observable Splitter")
        if uses_mitigation:
            if use_advanced:
                chain("Advanced Observable Mitigation")
            if use_noise_tailoring:
                chain("Noise Tailoring")
            if use_dd:
                chain("Dynamical Decoupling")
            if use_readout and not use_advanced:
                chain("Observable Readout Correction")
            chain("QWC Compute")
            if use_readout and use_advanced:
                chain("Advanced Readout Mitigation")
            if use_advanced:
                chain("Merge")
                G.add_edge("Advanced Observable Mitigation", "Merge")  # aux branch: scale_tasks → Merge
                arch_edges.add(("Advanced Observable Mitigation", "Merge"))
        else:
            chain("QWC Compute")
    else:
        if uses_mitigation:
            if use_advanced:
                chain("Advanced Distribution Mitigation")
            if use_noise_tailoring:
                chain("Noise Tailoring")
            if use_dd:
                chain("Dynamical Decoupling")
            if use_readout and not use_advanced:
                chain("Readout Mitigation")
            elif use_readout and use_advanced:
                chain("Advanced Readout Mitigation")
            if use_advanced:
                merge_source = "Advanced Distribution Mitigation"
            elif use_readout:
                merge_source = "Readout Mitigation"
            else:
                merge_source = None
            if merge_source:
                chain("Merge")
                G.add_edge(merge_source, "Merge")
                arch_edges.add((merge_source, "Merge"))

    chain(device)

    G.graph["device"] = device
    G.graph["arch_edges"] = arch_edges
    return G


_MITIGATION_NODES = {
    "Advanced Observable Mitigation",
    "Advanced Distribution Mitigation",
    "Noise Tailoring",
    "Dynamical Decoupling",
    "Advanced Readout Mitigation",
    "Observable Readout Correction",
    "Readout Mitigation",
}
_OBSERVABLE_NODES = {"Observable Splitter", "QWC Compute"}


def plot_run_job_graph(G, help: bool = False) -> None:
    """Render a run-job execution flow graph as an inline Jupyter image.

    Expects G.graph['device'] and G.graph['arch_edges'] as set by build_run_job_graph.
    """
    import io
    import base64
    import textwrap
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    device = G.graph.get("device", "Device")
    arch_edges: set = G.graph.get("arch_edges", set())

    # Haiqu brand palette
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    MORPHO_BLUE = "#093188"
    LIGHT_BLUE = "#C4D5FF"
    NEUTRAL_GREY = "#CCCCCC"
    ORANGE = "#FEA450"

    STYLES = {
        "input": {"fill": NEUTRAL_GREY, "stroke": "#888888", "text": BLACK},
        "layer": {"fill": MORPHO_BLUE, "stroke": MORPHO_BLUE, "text": WHITE},
        "observable": {"fill": LIGHT_BLUE, "stroke": MORPHO_BLUE, "text": MORPHO_BLUE},
        "mitigation": {"fill": LIGHT_BLUE, "stroke": MORPHO_BLUE, "text": MORPHO_BLUE},
        "merge": {"fill": ORANGE, "stroke": "#c07820", "text": BLACK},
        "device": {"fill": MORPHO_BLUE, "stroke": ORANGE, "text": WHITE},
    }
    ARROW_C = MORPHO_BLUE

    def _style(node):
        if node == "Input":
            return STYLES["input"]
        if node == device:
            return STYLES["device"]
        if node == "Merge":
            return STYLES["merge"]
        if node in _OBSERVABLE_NODES:
            return STYLES["observable"]
        if node in _MITIGATION_NODES:
            return STYLES["mitigation"]
        return STYLES["layer"]

    try:
        node_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        node_order = list(G.nodes())

    SPACING = 2.6
    NODE_W = 2.0
    NODE_H = 0.70
    n = len(node_order)

    fig, ax = plt.subplots(figsize=(max(6.0, SPACING * (n - 1) + 2.5), 2.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_axis_off()

    node_x = {node: i * SPACING for i, node in enumerate(node_order)}

    for node, x in node_x.items():
        style = _style(node)
        ax.add_patch(
            FancyBboxPatch(
                (x - NODE_W / 2, -NODE_H / 2),
                NODE_W,
                NODE_H,
                boxstyle="round,pad=0.05",
                linewidth=1.5,
                edgecolor=style["stroke"],
                facecolor=style["fill"],
                zorder=3,
            )
        )
        ax.text(
            x,
            0,
            textwrap.fill(node.upper(), width=13),
            ha="center",
            va="center",
            fontsize=7.5,
            color=style["text"],
            fontfamily="monospace",
            fontweight="bold",
            zorder=4,
            linespacing=1.3,
        )

    seen: set = set()
    skip_row: dict = {}
    for u, v in G.edges():
        xu, xv = node_x[u], node_x[v]
        gap = node_order.index(v) - node_order.index(u)
        is_arch = gap > 1 or (u, v) in arch_edges
        if (u, v) in seen:
            if is_arch:
                ax.annotate(
                    "",
                    xy=(xv - NODE_W / 2, 0.0),
                    xytext=(xu + NODE_W / 2, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.2, mutation_scale=10),
                    zorder=2,
                )
            continue
        seen.add((u, v))
        if not is_arch:
            ax.annotate(
                "",
                xy=(xv - NODE_W / 2, 0.0),
                xytext=(xu + NODE_W / 2, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.5, mutation_scale=12),
                zorder=2,
            )
        else:
            # Arch edges routed above nodes; each extra skip gets a higher lane
            lane = skip_row.get(gap, 0)
            skip_row[gap] = lane + 1
            y_off = NODE_H / 2 + 0.25 + lane * 0.30
            xs = [xu, xu, xv, xv]
            ys = [NODE_H / 2, y_off, y_off, NODE_H / 2]
            ax.plot(xs, ys, color=ARROW_C, lw=1.5, zorder=2)
            ax.annotate(
                "",
                xy=(xv, NODE_H / 2),
                xytext=(xv, y_off),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.5, mutation_scale=12),
                zorder=2,
            )

    ax.set_xlim(-SPACING * 0.5, SPACING * (n - 1) + SPACING * 0.5)
    ax.set_ylim(-0.85, 1.5)
    ax.margins(0)

    if help:
        handles, labels = [], []
        for node in node_order:
            desc = NODE_DESCRIPTIONS.get(node)
            if desc is None:
                continue
            style = _style(node)
            handles.append(Patch(facecolor=style["fill"], edgecolor=style["stroke"], linewidth=1.2))
            labels.append(f"{node.upper()}: {desc}")

        if handles:
            needed_w = max(len(label) for label in labels) * 0.054 + 1.0
            fig.set_figwidth(max(fig.get_figwidth(), needed_w))

            leg = ax.legend(
                handles,
                labels,
                title="EACH BOX IS A PROCESSING LAYER IN THE DATA FLOW FROM USER CIRCUITS TO DEVICE.",
                title_fontsize=7.5,
                fontsize=7.5,
                loc="upper left",
                bbox_to_anchor=(0.0, -0.08),
                framealpha=1.0,
                edgecolor=NEUTRAL_GREY,
                facecolor=WHITE,
                labelcolor=BLACK,
                handlelength=1.2,
                handleheight=1.0,
            )
            leg._legend_box.align = "left"
            leg.get_title().set_color("#555555")
            leg.get_title().set_fontfamily("monospace")
            for text in leg.get_texts():
                text.set_fontfamily("monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06, dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    html_str = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;" />'
    from haiqu.sdk.wiz.jupyter import render_template  # lazy to avoid circular import

    return display(HTML(render_template("RUN JOB EXECUTION FLOW", html_str)))


def draw_run_job(
    device_id: str,
    uses_observables: bool,
    uses_mitigation: bool,
    uses_packing: bool,
    uses_transpilation: bool = True,
    use_advanced: bool = True,
    use_noise_tailoring: bool = False,
    use_dd: bool = True,
    use_readout: bool = True,
    help: bool = False,
) -> None:
    """Render the execution flow graph for the run job."""
    G = build_run_job_graph(
        device_id,
        uses_observables,
        uses_mitigation,
        uses_packing,
        uses_transpilation,
        use_advanced,
        use_noise_tailoring,
        use_dd,
        use_readout,
    )
    return plot_run_job_graph(G, help=help)
