
from __future__ import annotations
import sys
from typing import List, Optional, Tuple
import cv2, numpy as np
import plotly.graph_objects as go
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FAKE_DECISION_BOUNDARY

_FAKE  = "#EF4444"
_REAL  = "#22C55E"
_BG    = "#0F1117"
_PANEL = "#1E2235"
_FONT  = {"color": "#E2E8F0"}
_GRID  = "#2D3555"


def frame_score_chart(verdict, height=380):
    scores  = verdict.frame_scores
    labels  = [f"Frame {s.frame_index}" for s in scores]
    probs   = [s.fake_prob for s in scores]
    colours = [_FAKE if p >= FAKE_DECISION_BOUNDARY else _REAL for p in probs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=probs, y=labels, orientation="h", marker_color=colours,
                         text=[f"{p:.1%}" for p in probs], textposition="outside",
                         hovertemplate="<b>%{y}</b><br>Fake prob: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=FAKE_DECISION_BOUNDARY, line_dash="dash", line_color="#F97316",
                  line_width=2, annotation_text="Decision", annotation_font_color="#F97316")
    fig.add_vline(x=verdict.weighted_score, line_dash="dot", line_color="#A78BFA",
                  line_width=2, annotation_text=f"Avg {verdict.weighted_score:.2%}",
                  annotation_position="bottom right", annotation_font_color="#A78BFA")
    fig.update_layout(title="Per-Frame Fake Probability", xaxis_title="Fake Probability",
                      xaxis_range=[0, 1.08], height=height, paper_bgcolor=_BG,
                      plot_bgcolor=_PANEL, font=_FONT, margin=dict(l=0,r=20,t=40,b=20),
                      xaxis=dict(gridcolor=_GRID, tickformat=".0%"),
                      yaxis=dict(gridcolor=_GRID))
    return fig


def score_gauge(verdict, height=280):
    score = verdict.weighted_score
    vc    = _FAKE if verdict.is_fake else _REAL
    label = f"{verdict.verdict}  ·  {verdict.confidence_tier} Confidence"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score*100,
        title={"text": label, "font": {"size": 16, "color": "#E2E8F0"}},
        number={"suffix": "%", "font": {"size": 36, "color": vc}},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#94A3B8"},
            "bar": {"color": vc, "thickness": 0.25},
            "bgcolor": _PANEL, "borderwidth": 1, "bordercolor": "#2D3555",
            "steps": [{"range":[0,35],"color":"#14532D"},{"range":[35,65],"color":"#78350F"},{"range":[65,100],"color":"#450A0A"}],
            "threshold": {"line":{"color":"#F97316","width":3},"thickness":0.8,"value":FAKE_DECISION_BOUNDARY*100},
        },
    ))
    fig.update_layout(height=height, paper_bgcolor=_BG, font=_FONT, margin=dict(l=10,r=10,t=30,b=10))
    return fig


def score_timeline(verdict, height=260):
    scores = sorted(verdict.frame_scores, key=lambda s: s.frame_index)
    xs = [s.frame_index for s in scores]
    ys = [s.fake_prob   for s in scores]
    cs = [s.certainty   for s in scores]
    upper = [min(1.0, y+(1-c)*0.15) for y,c in zip(ys,cs)]
    lower = [max(0.0, y-(1-c)*0.15) for y,c in zip(ys,cs)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs+xs[::-1], y=upper+lower[::-1], fill="toself",
                             fillcolor="rgba(167,139,250,0.15)", line={"width":0}, name="Uncertainty"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line={"color":"#A78BFA","width":2},
                             marker={"size":8,"color":[_FAKE if y>=FAKE_DECISION_BOUNDARY else _REAL for y in ys]},
                             name="Fake Prob", hovertemplate="Frame %{x}<br>Score: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=FAKE_DECISION_BOUNDARY, line_dash="dash", line_color="#F97316",
                  annotation_text="Decision", annotation_position="right", annotation_font_color="#F97316")
    fig.update_layout(title="Score Timeline", xaxis_title="Frame Index", yaxis_title="Fake Probability",
                      yaxis_range=[0,1], height=height, paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                      font=_FONT, margin=dict(l=0,r=80,t=40,b=20),
                      xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID, tickformat=".0%"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def build_rgb_thumbnail(bgr_frame, fake_prob, frame_idx, size=(224,224)):
    thumb = cv2.resize(bgr_frame, size, interpolation=cv2.INTER_AREA)
    if fake_prob is None:
        colour, label = (150,150,150), f"#{frame_idx} No face"
    elif fake_prob >= FAKE_DECISION_BOUNDARY:
        colour, label = (0,0,220), f"#{frame_idx} FAKE {fake_prob:.0%}"
    else:
        colour, label = (0,200,0), f"#{frame_idx} REAL {fake_prob:.0%}"
    b = 4
    thumb = cv2.copyMakeBorder(thumb, b,b,b,b, cv2.BORDER_CONSTANT, value=colour)
    cv2.rectangle(thumb, (0, thumb.shape[0]-28), (thumb.shape[1], thumb.shape[0]), (20,20,20), -1)
    cv2.putText(thumb, label, (6, thumb.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230,230,230), 1, cv2.LINE_AA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
