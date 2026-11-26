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
    top = (40 if title else 10) + (48 if legend else 0)
    bottom = max(10, fig.layout.margin.b or 0)     # keep room for animation controls
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=12), font=dict(size=15)) if title else None,
        height=height + (48 if legend else 0), margin=dict(l=10, r=10, t=top, b=bottom),
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
