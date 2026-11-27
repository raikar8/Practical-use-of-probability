"""Shared Streamlit / Plotly helpers so every page looks and behaves the same."""
from __future__ import annotations

import inspect

import plotly.graph_objects as go
import streamlit as st

# Categorical slots in fixed order (validated colour-blind-safe ordering).
# Truth is always neutral grey so the estimators carry the colour.
C = {
    "truth": "#8a8a8a",
    1: "#257de0",   # blue
    2: "#f47230",   # orange
    3: "#00b894",   # aqua
    4: "#f2a600",   # yellow
    5: "#ee7fac",   # magenta
    6: "#008a17",   # green
    7: "#6a4ce2",   # violet
    8: "#ec4647",   # red
}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, title: str | None = None, x: str | None = None, y: str | None = None,
          height: int = 380, legend: bool = True) -> go.Figure:
    """Consistent layout: title top-left, legend in its own band under the title, recessive grid."""
    named = sum(1 for tr in fig.data if tr.name and tr.showlegend is not False)
    legend_h = 22 * max(1, -(-named // 4)) + 4 if legend else 0      # ~4 entries per legend row
    top = (40 if title else 10) + legend_h
    bottom = max(10, fig.layout.margin.b or 0)     # keep room for animation controls
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=12), font=dict(size=15)) if title else None,
        height=height + legend_h, margin=dict(l=10, r=10, t=top, b=bottom),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(title=x, showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(title=y, showgrid=True, gridwidth=1, zeroline=False)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def takeaway(text: str) -> None:
    st.info(text, icon=":material/lightbulb:")


def break_it(text: str) -> None:
    st.warning(text, icon=":material/build:")


def in_practice(text: str) -> None:
    st.success(text, icon=":material/factory:")


def source(*objs) -> None:
    """Show the exact source code the page is running — no black boxes."""
    for obj in objs:
        mod = inspect.getmodule(obj).__name__
        st.caption(f"`{mod}.{obj.__name__}`")
        st.code(inspect.getsource(obj), language="python")


def play_controls(fig: go.Figure, frame_names: list[str], duration_ms: int, label: str = "t = ") -> go.Figure:
    """Play / pause buttons and a scrub slider for a Plotly animation (runs in the browser)."""
    anim = dict(frame=dict(duration=duration_ms, redraw=False), transition=dict(duration=0), fromcurrent=True,
                mode="immediate")
    fig.update_layout(
        margin=dict(b=110),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.13, xanchor="left", yanchor="top",
                          pad=dict(r=8, t=0), showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, anim]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate", transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.16, len=0.84, y=-0.10, yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(prefix=label, visible=True, xanchor="right"),
                      steps=[dict(label=nm, method="animate",
                                  args=[[nm], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                   transition=dict(duration=0))]) for nm in frame_names])],
    )
    return fig


def header(level: str, title: str, question: str) -> None:
    st.caption(level)
    st.title(title)
    st.markdown(f"*{question}*")


# ---------------------------------------------------------------------------
# Theory / Lab structure
# ---------------------------------------------------------------------------
# key: (section label, short title, theory page, lab page)
TOPICS = {
    "l0": ("Level 0", "Probability toolkit", "app_pages/theory/l0_probability.py", "app_pages/l0_probability.py"),
    "l1": ("Level 1", "Estimation: MLE, MAP, CRLB", "app_pages/theory/l1_estimation.py", "app_pages/l1_estimation.py"),
    "l2": ("Level 2", "Adaptive filters: LMS, NLMS, RLS", "app_pages/theory/l2_adaptive.py", "app_pages/l2_adaptive.py"),
    "l3": ("Level 3", "State-space models & the Kalman filter", "app_pages/theory/l3_kalman.py",
           "app_pages/l3_kalman.py"),
    "l4": ("Level 4", "Hidden Markov models", "app_pages/theory/l4_hmm.py", "app_pages/l4_hmm.py"),
    "l5": ("Level 5", "Particle filters, IMM, change points", "app_pages/theory/l5_advanced.py",
           "app_pages/l5_advanced.py"),
    "vehicle": ("Domain", "Car GPS + IMU fusion", "app_pages/theory/d_vehicle.py", "app_pages/d_vehicle.py"),
    "netcode": ("Domain", "Online game lag prediction", "app_pages/theory/d_netcode.py", "app_pages/d_netcode.py"),
    "speech": ("Domain", "Voice activity detection", "app_pages/theory/d_speech.py", "app_pages/d_speech.py"),
    "enhance": ("Domain", "Noise suppression & hearing aids", "app_pages/theory/d_enhance.py",
                "app_pages/d_enhance.py"),
    "markets": ("Domain", "Market regimes & volatility", "app_pages/theory/d_markets.py", "app_pages/d_markets.py"),
    "sports": ("Domain", "Sports ratings & win probability", "app_pages/theory/d_sports.py", "app_pages/d_sports.py"),
}


def eq(latex: str, number: int | str | None = None) -> None:
    """A display equation, optionally numbered so the text and the Lab can refer to it."""
    # KaTeX's \tag is placed relative to a shrink-wrapped box in st.latex and overlaps wide equations,
    # so the number is set inline, a fixed gap to the right of the equation.
    st.latex(latex + (rf" \qquad\qquad \text{{({number})}}" if number is not None else ""))


def theory_header(key: str, why: str) -> None:
    level, title, _, lab = TOPICS[key]
    st.caption(f"{level} · Theory")
    st.title(title)
    st.markdown(why)
    st.page_link(lab, label="Skip to the Lab", icon=":material/science:")


def section(title: str) -> None:
    st.divider()
    st.header(title)


def lab_intro(key: str, goal: str, on_screen: list[str], start_here: str) -> None:
    """Prologue for every Lab: what we're doing, what's on screen, where to begin."""
    level, title, theory, _ = TOPICS[key]
    st.caption(f"{level} · Lab")
    st.title(title)
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**What this lab is for.** {goal}")
            st.markdown("**What you're looking at**\n" + "\n".join(f"- {x}" for x in on_screen))
            st.markdown(f"**Start here.** {start_here}")
        with c2:
            st.page_link(theory, label="Read the theory first", icon=":material/menu_book:")
            st.caption("Every control has a **?** tooltip. 'Controls explained' and 'Guided experiments' are below "
                       "each panel.")


def controls(rows: list[tuple[str, str, str, str]], title: str = "Controls explained") -> None:
    """rows: (control, symbol in the theory, what it means, what happens when you increase it)."""
    with st.expander(title, icon=":material/tune:"):
        md = ["| Control | Symbol | What it means | Turn it up and… |", "|---|---|---|---|"]
        md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
        st.markdown("\n".join(md))


def experiments(items: list[tuple[str, str, str]], title: str = "Guided experiments") -> None:
    """items: (what to do, what to watch, why it happens)."""
    with st.container(border=True):
        st.markdown(f"**:material/science: {title}**")
        for i, (do, watch, why) in enumerate(items, 1):
            st.markdown(f"**{i}. {do}**  \n*Watch:* {watch}  \n*Why:* {why}")


def check(items: list[tuple[str, str]]) -> None:
    """Self-check questions with hidden answers."""
    for q, a in items:
        with st.expander(f"❓ {q}"):
            st.markdown(a)


def lab_map(key: str, rows: list[tuple[str, str]]) -> None:
    """How the theory shows up in the Lab: (thing in the Lab, what it is in the theory)."""
    md = ["| In the Lab | In the theory |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    st.markdown("\n".join(md))
    st.page_link(TOPICS[key][3], label="Open the Lab", icon=":material/science:")


def real_audio(key: str, max_seconds: float = 60.0):
    """Record from the browser microphone or upload a WAV. Returns mono float audio at 16 kHz, or None."""
    from ssp.sims.speech import load_wav
    how = st.radio("Real audio", ["Record with your microphone", "Upload a WAV"], key=f"{key}_how",
                   horizontal=True)
    if how.startswith("Record"):
        label = "Press the mic, allow access when the browser asks, then press stop"
        try:
            data = st.audio_input(label, sample_rate=16000, key=f"{key}_rec")
        except TypeError:                       # older Streamlit without sample_rate
            data = st.audio_input(label, key=f"{key}_rec")
        st.caption("Tip: stay quiet for the first second, then talk with a few pauses. A fan, traffic, a running "
                   "tap or music in the background makes it interesting. Browsers may apply their own noise "
                   "suppression to the mic; uploading a WAV from a recorder avoids that.")
    else:
        data = st.file_uploader("WAV file (mono or stereo, any rate)", type=["wav"], key=f"{key}_up")
    if data is None:
        return None
    x = load_wav(data.getvalue(), max_seconds)
    if x.size < 16000:
        st.warning("That's under a second of audio. Record a few seconds with some pauses.")
        return None
    return x
