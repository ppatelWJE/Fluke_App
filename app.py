# app_test_13.py
# Streamlit app for plotting utility trends with export features (updated)
# Changes from app_test_12:
# - Added an "Export text sizes" section with explicit inputs for Title, Axis titles, Ticks, Legend, and Subplot titles
# - Defaults: Title 50 pt; Axis titles 28 pt; Ticks 18 pt; Legend 24 pt; Subplot titles 35 pt
# - Tick length fixed to 6 for clearer marks in exports

import io
from datetime import datetime, timedelta
import re
from typing import List, Optional, Tuple, Dict, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Fluke Recordings", layout="wide")
st.title("Fluke Recordings Plotter and Exporter")

# ---------------------------
# Config
# ---------------------------
START_COL = "Start(Eastern Standard Time)"
STOP_COL = "Stop(Eastern Standard Time)"

DEFAULT_PALETTE = (
    px.colors.qualitative.D3
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Dark24
)

def ensure_hex(color: str) -> Optional[str]:
    if isinstance(color, str) and color.startswith("#") and len(color) in (4, 7):
        return color
    return None

def to_pydt(x):
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return x

def clamp_dt(v: datetime, lo: datetime, hi: datetime) -> datetime:
    if v < lo: return lo
    if v > hi: return hi
    return v

@st.cache_data
def load_log(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=",", on_bad_lines="skip", low_memory=False)
    for col in (START_COL, STOP_COL):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
    if START_COL in df.columns:
        df = df.dropna(subset=[START_COL])
        df = df.sort_values(by=START_COL).reset_index(drop=True)
    return df

def slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "export"

def to_seconds(series: pd.Series, x_is_time: bool, start_bound) -> pd.Series:
    if x_is_time:
        return (series - pd.to_datetime(start_bound)).dt.total_seconds()
    else:
        return series.astype(float) - float(start_bound)


def make_stacked_figure(
    x_sec: pd.Series,
    df: pd.DataFrame,
    y_series: List[str],
    global_title: str,
    y_label: Union[str, None] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    *,
    line_color: str = "black",
    y_titles: Optional[List[str]] = None,
    y_bounds_per_series: Optional[Dict[str, Tuple[float, float]]] = None,
    y_label_per_series: Optional[Dict[str, str]] = None,
    # Reference line options
    ref_value: Optional[float] = None,
    ref_label: Optional[str] = None,
    ref_series_only: Optional[List[str]] = None,  # if provided, draw ref line only on these series
    ref_color: str = "red",
    ref_dash: str = "dash",
    # Export styling sizes
    title_size: int = 50,            # defaults will be overridden by UI inputs
    base_font_size: int = 18,
    axis_title_size: int = 28,
    tick_font_size: int = 18,
    legend_font_size: int = 24,
    subtitle_size: int = 35,
    tick_len: int = 6,
    add_subplot_borders: bool = True,
):
    """Build stacked figure with export-friendly styling and optional per-series Y labels."""
    rows = len(y_series)
    subtitles = y_titles if (y_titles and len(y_titles) == rows) else y_series
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=subtitles if rows > 1 else None,
    )

    for i, y in enumerate(y_series, start=1):
        fig.add_trace(
            go.Scatter(
                x=x_sec,
                y=df[y],
                mode="lines",
                name=y,
                line=dict(width=2, color=line_color),
                showlegend=False if ref_value is not None else (rows == 1),
                hovertemplate=f"<b>{subtitles[i-1]}</b><br>Time (sec): %{{x}}<br>{y}: %{{y}}<extra></extra>",
            ), row=i, col=1
        )

        # Determine bounds for this series
        bounds = None
        if y_bounds_per_series and y in y_bounds_per_series:
            bounds = y_bounds_per_series[y]
        elif y_bounds is not None:
            bounds = y_bounds

        # Apply Y label & bounds
        label_for_row = None
        if y_label_per_series and y in y_label_per_series:
            label_for_row = y_label_per_series[y]
        else:
            label_for_row = y_label if y_label is not None else None

        if label_for_row is not None:
            fig.update_yaxes(title_text=label_for_row, row=i, col=1)
        if bounds is not None:
            ymin, ymax = bounds
            if ymin == ymax:
                pad = max(1.0, abs(ymin) * 0.01)
                ymin, ymax = ymin - pad, ymax + pad
            fig.update_yaxes(range=[ymin, ymax], row=i, col=1)
        fig.update_xaxes(showticklabels=True, title_text="Time (sec)", row=i, col=1)

        # Reference line per subplot
        if ref_value is not None:
            if (ref_series_only is None) or (y in (ref_series_only or [])):
                fig.add_hline(y=float(ref_value), line_dash=ref_dash, line_color=ref_color, line_width=2, row=i, col=1)

    # Legend entry for reference line using a trace that is legend-only
    if ref_value is not None and ref_label:
        if len(x_sec) >= 2:
            x0, x1 = x_sec.iloc[0], x_sec.iloc[-1]
        else:
            x0, x1 = 0, 1
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[ref_value, ref_value], mode="lines",
                line=dict(color=ref_color, dash=ref_dash, width=2),
                name=ref_label, hoverinfo="skip", showlegend=True, visible='legendonly',
            ), row=1, col=1
        )

    GRID_COLOR = "#B5B5B5"
    GRID_WIDTH = 1.2
    fig.update_layout(
        font=dict(color="black", size=base_font_size),
        paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text=global_title, x=0.5, xanchor="center", font=dict(size=title_size, color="black")),
        template="plotly_white",
        showlegend=True if (ref_value is not None and ref_label) else (False if rows > 1 else True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            font=dict(color="black", size=legend_font_size),
            bgcolor="white", bordercolor="black", borderwidth=1
        ),
        margin=dict(l=10, r=10, t=100, b=10), height=max(380, 300 * rows),
    )

    # Subplot title font to requested size
    for ann in fig.layout.annotations or []:
        ann.font = dict(size=subtitle_size, color="black")

    # Grid + ticks styling
    fig.for_each_xaxis(lambda ax: ax.update(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=GRID_WIDTH, zeroline=False, showline=False,
        ticks="outside", tickcolor="#000", ticklen=tick_len, tickfont=dict(color="black", size=tick_font_size),
        title_font=dict(size=axis_title_size, color="black"),
    ))
    fig.for_each_yaxis(lambda ay: ay.update(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=GRID_WIDTH, zeroline=False, showline=False,
        tickfont=dict(color="black", size=tick_font_size), title_font=dict(size=axis_title_size, color="black"),
    ))

    # Subplot borders
    if add_subplot_borders:
        try:
            xdom = fig.layout.xaxis.domain
        except Exception:
            xdom = [0.0, 1.0]
        for i in range(1, rows + 1):
            yaxis_name = 'yaxis' if i == 1 else f'yaxis{i}'
            ydom = getattr(fig.layout, yaxis_name).domain
            fig.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=xdom[0], x1=xdom[1], y0=ydom[0], y1=ydom[1],
                line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)", layer="above"
            )

    return fig


def compute_letter_export_size(n_rows: int, dpi: int = 300) -> Tuple[int, int]:
    usable_w_in, usable_h_in = 6.5, 9.0
    width_px = int(usable_w_in * dpi)
    max_h_px = int(usable_h_in * dpi)
    reserved_title_px = 180
    available_px = max_h_px - reserved_title_px
    per_row_px_min = 250
    per_row_px = max(per_row_px_min, int(available_px / max(n_rows, 1)))
    height_px = reserved_title_px + per_row_px * max(n_rows, 1)
    height_px = min(height_px, max_h_px)
    return width_px, height_px

# ---------------------------
# Sidebar: Upload + Export toggles
# ---------------------------
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Upload Fluke exported trend file (.txt or .csv) \nMake sure exported file uses COMMA as delimiter",
        type=["txt", "csv"],
        help="Comma-delimited file with header row (e.g., Start/Stop timestamps and metrics).",
    )
    st.header("Export enable")
    export_png_enabled = st.checkbox("Enable PNG export (requires kaleido)", value=True)

if uploaded is None:
    st.info(
        f"👆 Upload your log file to begin. Expected columns include `{START_COL}`, `{STOP_COL}`, and numeric metrics."
    )
    st.stop()

df = load_log(uploaded.getvalue())
if df.empty:
    st.error("No data found after parsing. Check delimiter and header row.")
    st.stop()

#Header title
st.header("Plotting")

# ---------------------------
# X-axis selection (MAIN screen)
# ---------------------------
with st.expander("Set X-axis", expanded=False):
    x_mode = st.radio(
        "Choose X‑axis",
        ["Start(Eastern Standard Time)", "Elapsed seconds (0,1,2,…)"]
        , index=0,
        help="Use actual timestamps, or a synthetic 1-second tick starting at 0.",
    )

    if x_mode.startswith("Elapsed"):
        if START_COL in df.columns:
            df = df.sort_values(by=START_COL).reset_index(drop=True)
        df["Elapsed_s"] = range(len(df))
        x_col = "Elapsed_s"
        x_is_time = False
    else:
        x_col = START_COL if START_COL in df.columns else df.columns[0]
        x_is_time = pd.api.types.is_datetime64_any_dtype(df[x_col])
        if not x_is_time:
            st.warning(
                f"Selected timestamp X-axis requires '{x_col}' to be datetime. Switch to 'Elapsed seconds' if needed."
            )

# ---------------------------
# Series selection (on-screen chart)
# ---------------------------
with st.expander("Series Selection", expanded=True):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    y_candidates = [c for c in numeric_cols if c != x_col]

    # Default selection: PowerP_Total_avg if present, else first available numeric
    _default_pref = ["PowerP_Total_avg"]
    _default_ys = [c for c in _default_pref if c in y_candidates] or (y_candidates[:1] if y_candidates else [])

    # Keep selection in session state so group buttons can override it
    if "y_sel" not in st.session_state:
        st.session_state["y_sel"] = _default_ys

    # Group buttons (Voltage, Current, Total Power)
    cgb1, cgb2, cgb3, _ = st.columns([1.1, 1.1, 1.2, 2])
    with cgb1:
        if st.button("Group: Voltage (Vab, Vca, Vbc)"):
            target = [s for s in ["Vrms_AB_avg", "Vrms_CA_avg", "Vrms_BC_avg"] if s in y_candidates]
            st.session_state["y_sel"] = target or st.session_state["y_sel"]
            st.rerun()
    with cgb2:
        if st.button("Group: Current (Ia, Ib, Ic)"):
            target = [s for s in ["Irms_A_avg", "Irms_B_avg", "Irms_C_avg"] if s in y_candidates]
            st.session_state["y_sel"] = target or st.session_state["y_sel"]
            st.rerun()
    with cgb3:
        if st.button("Group: Total Power (P_total)"):
            target = [s for s in ["PowerP_Total_avg"] if s in y_candidates]
            st.session_state["y_sel"] = target or st.session_state["y_sel"]
            st.rerun()

    # Multiselect bound to state
    y_cols = st.multiselect(
        "Y series (one or more)",
        options=y_candidates,
        default=st.session_state["y_sel"],
        key="y_sel"
    )

    if not y_cols:
        st.warning("Select at least one numeric Y series.")
        st.stop()

# ---------------------------
# Range controls
# ---------------------------
with st.expander("Range Selection", expanded=False):
    if x_is_time:
        xmin_pd = pd.to_datetime(df[x_col].min(), errors="coerce")
        xmax_pd = pd.to_datetime(df[x_col].max(), errors="coerce")
        if pd.isna(xmin_pd) or pd.isna(xmax_pd):
            st.error("No valid timestamps in the selected X column.")
            st.stop()
        xmin = to_pydt(xmin_pd); xmax = to_pydt(xmax_pd)
        if "ts_range" not in st.session_state:
            st.session_state.ts_range = (xmin, xmax)
        s0, e0 = st.session_state.ts_range
        s0 = to_pydt(s0) if s0 else xmin
        e0 = to_pydt(e0) if e0 else xmax
        s0 = clamp_dt(s0, xmin, xmax); e0 = clamp_dt(e0, xmin, xmax)
        if s0 > e0: s0, e0 = e0, s0
        st.session_state.ts_range = (s0, e0)

        c1, c2, c3 = st.columns([1.2, 1.2, 0.7])
        with c1:
            ts_start_str = st.text_input("Start timestamp", value=s0.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                        help="Example: 2025-10-17 11:55:00.000")
        with c2:
            ts_end_str = st.text_input("End timestamp", value=e0.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    help="Example: 2025-10-17 11:56:00.000")
        with c3:
            if st.button("Apply typed range"):
                new_s = pd.to_datetime(ts_start_str, errors="coerce")
                new_e = pd.to_datetime(ts_end_str, errors="coerce")
                if pd.isna(new_s) or pd.isna(new_e):
                    st.warning("Could not parse the typed timestamps.")
                else:
                    ns = to_pydt(new_s); ne = to_pydt(new_e)
                    ns = clamp_dt(ns, xmin, xmax); ne = clamp_dt(ne, xmin, xmax)
                    if ns > ne: ns, ne = ne, ns
                    st.session_state.ts_range = (ns, ne)
                    st.rerun()

        slider_val = st.slider("Drag to set range (timestamps)", min_value=xmin, max_value=xmax,
                            value=st.session_state.ts_range, step=timedelta(seconds=1),
                            help="This range drives the plot and exports.")
        start_bound, end_bound = slider_val
    else:
        xmin = int(df[x_col].min()); xmax = int(df[x_col].max())
        if "num_range" not in st.session_state:
            st.session_state.num_range = (xmin, xmax)
        n0, n1 = st.session_state.num_range
        if n0 > n1: n0, n1 = n1, n0
        c1, c2, c3 = st.columns([1, 1, 0.7])
        with c1:
            n_start = st.number_input("Start (sec)", value=int(n0), step=1)
        with c2:
            n_end = st.number_input("End (sec)", value=int(n1), step=1)
        with c3:
            if st.button("Apply typed range"):
                s = int(n_start); e = int(n_end)
                if s > e: s, e = e, s
                st.session_state.num_range = (s, e)
                st.rerun()

        slider_val = st.slider("Drag to set range (seconds)", min_value=xmin, max_value=xmax,
                            value=st.session_state.num_range, step=1,
                            help="This range drives the plot and exports.")
        start_bound, end_bound = slider_val

# ---------------------------
# Color Selection
# ---------------------------
with st.expander("Colors Selection", expanded=False):
    if "series_colors" not in st.session_state:
        st.session_state.series_colors = {}

    palette_iter = iter(DEFAULT_PALETTE)
    color_map = {}
    cols = st.columns(min(4, max(1, len(y_cols))))
    for i, y in enumerate(y_cols):
        default_color = st.session_state.series_colors.get(y)
        if not default_color:
            try:
                default_color = next(palette_iter)
            except StopIteration:
                default_color = "#1f77b4"
        with cols[i % len(cols)]:
            picked = st.color_picker(f"{y}", value=default_color)
            final = ensure_hex(picked) or default_color
            st.session_state.series_colors[y] = final
            color_map[y] = final

# ---------------------------
# Chart Options Selection
# ---------------------------
with st.expander("Chart Options", expanded=False):
    title = st.text_input("Title", value="Fluke Recording Data")
    x_title = st.text_input("X‑axis label", value=x_col)
    y_title = st.text_input("Y‑axis label", value="Value")

plot_df = df[(df[x_col] >= start_bound) & (df[x_col] <= end_bound)].copy()
if pd.api.types.is_numeric_dtype(plot_df[x_col]) or pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
    plot_df = plot_df.sort_values(by=x_col)

if plot_df.empty:
    st.warning("No data rows within the selected range.")
    st.stop()

fig = go.Figure()
for y in y_cols:
    fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y], mode="lines", name=y,
                             line=dict(color=color_map.get(y), width=2, shape="linear"),
                             hovertemplate=f"<b>{y}</b><br>{x_col}: %{{x}}<br>{y}: %{{y}}<extra></extra>"))

fig.update_layout(title=title, template="plotly_white",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                              font=dict(color="black"), bgcolor="white", bordercolor="black", borderwidth=1),
                  margin=dict(l=10, r=10, t=60, b=10))
fig.update_xaxes(title=x_title, showgrid=True, gridcolor="#E5E5E5", gridwidth=0.6, range=[start_bound, end_bound],
                 rangeslider_visible=bool(pd.api.types.is_datetime64_any_dtype(plot_df[x_col])))
fig.update_yaxes(title=y_title, showgrid=True, gridcolor="#E5E5E5", gridwidth=0.6)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Exporting
# ---------------------------
st.subheader("Exporting")

# New: explicit text size controls for exports
with st.expander("Export text sizes (applies to all exports)", expanded=False):
    c1, c2, c3 = st.columns([1,1,1])
    c4, c5 = st.columns([1,1])
    title_size = c1.number_input("Title (pt)", min_value=8, max_value=120, value=50, step=1)
    axis_title_size = c2.number_input("Axis titles (pt)", min_value=8, max_value=120, value=28, step=1)
    tick_font_size = c3.number_input("Ticks (pt)", min_value=6, max_value=120, value=18, step=1,
                                     help="Tick label font size. Tick mark length is fixed to 6 for clarity.")
    legend_font_size = c4.number_input("Legend (pt)", min_value=8, max_value=120, value=24, step=1)
    subtitle_size = c5.number_input("Subplot titles (pt)", min_value=8, max_value=120, value=35, step=1)

x_sec = to_seconds(plot_df[x_col], x_is_time, start_bound)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
voltage_candidates = [c for c in numeric_cols if c.lower().startswith("vrms")]
current_candidates = [c for c in numeric_cols if c.lower().startswith("irms")]
# Include PfCapInd_Total_avg in power defaults/candidates
power_candidates = list({
    *[c for c in numeric_cols if c.lower().startswith("power")],
    *( ["PfCapInd_Total_avg"] if "PfCapInd_Total_avg" in numeric_cols else [] ),
})

# ---- Voltage export ----
with st.expander("Voltage", expanded=False):
    default_voltage = [c for c in ["Vrms_AB_avg", "Vrms_BC_avg", "Vrms_CA_avg"] if c in voltage_candidates]
    volt_sel = st.multiselect(
        "Select voltage Y series to export",
        options=voltage_candidates,
        default=default_voltage,
        help="Each selected series will be placed on its own stacked subplot in one PNG.",
        key="volt_sel",
    )

    # Bounds controls (0..1000 slider) + typed inputs remain unconstrained
    if "volt_bounds" not in st.session_state:
        st.session_state.volt_bounds = (0.0, 1000.0)
    lb, ub = st.session_state.volt_bounds
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        volt_min_in = st.number_input("Voltage lower bound (V)", value=float(lb), step=0.1, format="%.3f")
    with c2:
        volt_max_in = st.number_input("Voltage upper bound (V)", value=float(ub), step=0.1, format="%.3f")
    with c3:
        if st.button("Apply typed bounds", key="apply_volt_bounds"):
            nl = float(volt_min_in); nu = float(volt_max_in)
            if nl > nu: nl, nu = nu, nl
            st.session_state.volt_bounds = (nl, nu)
            st.rerun()

    s_lb, s_ub = st.session_state.volt_bounds
    step = max((1000.0 - 0.0) / 500.0, 0.001)
    st.slider("Voltage bounds (V)", min_value=0.0, max_value=1000.0,
              value=(max(0.0, float(s_lb)), min(1000.0, float(s_ub))),
              step=float(step), key="volt_bounds",
              help="Drag to adjust export Y-limits (0 to 1000 V).")

    nominal_text = st.text_input("Nominal Voltage (V) [optional]", value="", placeholder="e.g., 480",
                                 help="If provided, a dashed red line is drawn at this value on each Voltage subplot.")
    nominal_value = None
    if nominal_text.strip():
        try:
            nominal_value = float(nominal_text)
        except ValueError:
            st.warning("Nominal Voltage must be numeric (e.g., 480). Ignoring for now.")

    volt_title = st.text_input("Export title (Voltage)", value="Voltage Export")
    if st.button("Generate Voltage PNG"):
        if not volt_sel:
            st.warning("Select at least one voltage series.")
        else:
            title_map = {"Vrms_AB_avg": "Voltage Phase A-B", "Vrms_BC_avg": "Voltage Phase B-C", "Vrms_CA_avg": "Voltage Phase C-A"}
            y_titles = [title_map.get(y, y) for y in volt_sel]
            vmin, vmax = st.session_state.volt_bounds
            fig_volt = make_stacked_figure(
                x_sec=x_sec, df=plot_df, y_series=volt_sel, global_title=volt_title,
                y_label="Voltage (V)",
                y_bounds=(vmin, vmax), line_color="black", y_titles=y_titles,
                ref_value=nominal_value, ref_label=("Nominal voltage" if nominal_value is not None else None),
                ref_series_only=None,
                # Text sizes from the new controls
                title_size=title_size, base_font_size=max(10, int(0.36*axis_title_size + tick_font_size*0.2)),
                axis_title_size=axis_title_size, tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                subtitle_size=subtitle_size, tick_len=6, add_subplot_borders=True,
            )
            width_px, height_px = compute_letter_export_size(len(volt_sel), dpi=300)
            try:
                png_bytes = fig_volt.to_image(format="png", width=width_px, height=height_px, scale=1)
                fname = f"{slugify(volt_title)}_voltage.png"
                st.download_button("⬇️ Download Voltage PNG", data=png_bytes, file_name=fname, mime="image/png")
            except Exception:
                st.error("PNG export needs 'kaleido'. Install it with: pip install kaleido")

# ---- Current export ----
with st.expander("Current", expanded=False):
    curr_sel = st.multiselect(
        "Select current Y series to export",
        options=current_candidates,
        default=[c for c in ["Irms_A_avg", "Irms_B_avg", "Irms_C_avg"] if c in current_candidates],
        help="Each selected series will be placed on its own stacked subplot in one PNG.",
        key="curr_sel",
    )

    if "curr_bounds" not in st.session_state:
        st.session_state.curr_bounds = (0.0, 2000.0)
    lb, ub = st.session_state.curr_bounds
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        curr_min_in = st.number_input("Current lower bound (A)", value=float(lb), step=0.1, format="%.3f")
    with c2:
        curr_max_in = st.number_input("Current upper bound (A)", value=float(ub), step=0.1, format="%.3f")
    with c3:
        if st.button("Apply typed bounds", key="apply_curr_bounds"):
            nl = float(curr_min_in); nu = float(curr_max_in)
            if nl > nu: nl, nu = nu, nl
            st.session_state.curr_bounds = (nl, nu)
            st.rerun()

    s_lb, s_ub = st.session_state.curr_bounds
    step = max((2000.0 - 0.0) / 500.0, 0.001)
    st.slider("Current bounds (A)", min_value=0.0, max_value=2000.0,
              value=(max(0.0, float(s_lb)), min(2000.0, float(s_ub))),
              step=float(step), key="curr_bounds",
              help="Drag to adjust export Y-limits (0 to 2000 A).")

    # FLA input
    fla_text = st.text_input("FLA (A) [optional]", value="", placeholder="e.g., 52.3",
                             help="If provided, a dashed red line is drawn at this value on each Current subplot.")
    fla_value = None
    if fla_text.strip():
        try:
            fla_value = float(fla_text)
        except ValueError:
            st.warning("FLA must be numeric (e.g., 52.3). Ignoring for now.")

    curr_title = st.text_input("Export title (Current)", value="Current Export")
    if st.button("Generate Current PNG"):
        if not curr_sel:
            st.warning("Select at least one current series.")
        else:
            cmin, cmax = st.session_state.curr_bounds
            curr_title_map = {"Irms_A_avg": "Phase A Current", "Irms_B_avg": "Phase B Current", "Irms_C_avg": "Phase C Current"}
            curr_y_titles = [curr_title_map.get(y, y) for y in curr_sel]
            fig_curr = make_stacked_figure(
                x_sec=x_sec, df=plot_df, y_series=curr_sel, global_title=curr_title,
                y_label="Current (A)",
                y_bounds=(cmin, cmax), line_color="black", y_titles=curr_y_titles,
                ref_value=fla_value, ref_label=("FLA" if fla_value is not None else None),
                ref_series_only=None,
                title_size=title_size, base_font_size=max(10, int(0.36*axis_title_size + tick_font_size*0.2)),
                axis_title_size=axis_title_size, tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                subtitle_size=subtitle_size, tick_len=6, add_subplot_borders=True,
            )
            width_px, height_px = compute_letter_export_size(len(curr_sel), dpi=300)
            try:
                png_bytes = fig_curr.to_image(format="png", width=width_px, height=height_px, scale=1)
                fname = f"{slugify(curr_title)}_current.png"
                st.download_button("⬇️ Download Current PNG", data=png_bytes, file_name=fname, mime="image/png")
            except Exception:
                st.error("PNG export needs 'kaleido'. Install it with: pip install kaleido")

# ---- Power export ----
with st.expander("Power", expanded=False):
    pwr_sel = st.multiselect(
        "Select power Y series to export",
        options=power_candidates,
        default=[c for c in ["PowerP_Total_avg", "PfCapInd_Total_avg"] if c in power_candidates],
        help="Each selected series will be placed on its own stacked subplot in one PNG.",
        key="pwr_sel",
    )

    # Global power bounds (-10000..300000) plus PF bounds (0..1) if PF included
    if "pwr_bounds" not in st.session_state:
        st.session_state.pwr_bounds = (-10000.0, 300000.0)
    if "pf_bounds" not in st.session_state:
        st.session_state.pf_bounds = (0.0, 1.0)

    # Power (W) bounds typed + slider
    lb, ub = st.session_state.pwr_bounds
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        pwr_min_in = st.number_input("Power lower bound (W)", value=float(lb), step=1.0, format="%.3f")
    with c2:
        pwr_max_in = st.number_input("Power upper bound (W)", value=float(ub), step=1.0, format="%.3f")
    with c3:
        if st.button("Apply typed bounds", key="apply_pwr_bounds"):
            nl = float(pwr_min_in); nu = float(pwr_max_in)
            if nl > nu: nl, nu = nu, nl
            st.session_state.pwr_bounds = (nl, nu)
            st.rerun()

    s_lb, s_ub = st.session_state.pwr_bounds
    step = max((300000.0 - (-10000.0)) / 500.0, 1.0)
    st.slider("Power bounds (W)", min_value=-10000.0, max_value=300000.0,
              value=(max(-10000.0, float(s_lb)), min(300000.0, float(s_ub))),
              step=float(step), key="pwr_bounds",
              help="Drag to adjust export Y-limits for power (-10k to 30k W).")

    # Additional PF slider if PfCapInd_Total_avg is in selection
    if "PfCapInd_Total_avg" in pwr_sel:
        pf_lb, pf_ub = st.session_state.pf_bounds
        st.slider("Power Factor bounds (PF)", min_value=0.0, max_value=1.0,
                  value=(max(0.0, float(pf_lb)), min(1.0, float(pf_ub))), step=0.001,
                  key="pf_bounds", help="Adjust bounds for Power Factor subplot (0 to 1).")

    # Rated Power input (applies ONLY to PowerP_Total_avg)
    rated_text = st.text_input("Rated Power (W) [optional]", value="", placeholder="e.g., 250000",
                               help="If provided, a dashed red line is drawn ONLY on the PowerP_Total_avg subplot.")
    rated_value = None
    if rated_text.strip():
        try:
            rated_value = float(rated_text)
        except ValueError:
            st.warning("Rated Power must be numeric (e.g., 250000). Ignoring for now.")

    pwr_title = st.text_input("Export title (Power)", value="Power Export")
    if st.button("Generate Power PNG"):
        if not pwr_sel:
            st.warning("Select at least one power series.")
        else:
            pmin, pmax = st.session_state.pwr_bounds
            # Per-series titles for power plots
            pwr_title_map = {"PowerP_Total_avg": "Active Power", "PfCapInd_Total_avg": "Power Factor"}
            pwr_y_titles = [pwr_title_map.get(y, y) for y in pwr_sel]
            # Per-series bounds (use PF bounds only for PF subplot)
            y_bounds_overrides = {}
            if "PfCapInd_Total_avg" in pwr_sel:
                pf_lb, pf_ub = st.session_state.pf_bounds
                y_bounds_overrides["PfCapInd_Total_avg"] = (float(pf_lb), float(pf_ub))

            # Per-series Y label override for PF subplot
            y_labels_overrides = {"PfCapInd_Total_avg": "Power Factor (ABS)"}

            fig_pwr = make_stacked_figure(
                x_sec=x_sec, df=plot_df, y_series=pwr_sel, global_title=pwr_title,
                y_label="Power (W)",
                y_bounds=(pmin, pmax), line_color="black", y_titles=pwr_y_titles,
                y_bounds_per_series=y_bounds_overrides,
                y_label_per_series=y_labels_overrides,
                ref_value=rated_value, ref_label=("Rated Power" if rated_value is not None else None),
                ref_series_only=["PowerP_Total_avg"],
                title_size=title_size, base_font_size=max(10, int(0.36*axis_title_size + tick_font_size*0.2)),
                axis_title_size=axis_title_size, tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                subtitle_size=subtitle_size, tick_len=6, add_subplot_borders=True,
            )
            width_px, height_px = compute_letter_export_size(len(pwr_sel), dpi=300)
            try:
                png_bytes = fig_pwr.to_image(format="png", width=width_px, height=height_px, scale=1)
                fname = f"{slugify(pwr_title)}_power.png"
                st.download_button("⬇️ Download Power PNG", data=png_bytes, file_name=fname, mime="image/png")
            except Exception:
                st.error("PNG export needs 'kaleido'. Install it with: pip install kaleido")

# ---------------------------
# Generic on-screen chart exports
# ---------------------------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button("⬇️ Download HTML (interactive)", data=html, file_name="trend_plot.html", mime="text/html")
with col2:
    if export_png_enabled:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button("⬇️ Download PNG (current on-screen chart)", data=png_bytes,
                               file_name="trend_plot.png", mime="image/png")
        except Exception:
            st.caption("To enable PNG export, install **kaleido**: `pip install kaleido`")
