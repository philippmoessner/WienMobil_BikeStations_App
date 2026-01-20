# ============================================================
# Vienna Bike Stations – Cluster Explorer (Communication Tool)
# ============================================================

import os
import json
import pickle
import re
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import folium
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="WienMobil Bicycle Stations – Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    with open("station_information.json", "r") as f:
        j = json.load(f)
    stations = pd.DataFrame(j["data"]["stations"]).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.set_index("station_id")

    if os.path.exists("data.pickle"):
        with open("data.pickle", "rb") as f:
            dfs = pickle.load(f)
        dfs = {str(k): v for k, v in dfs.items()}
    else:
        dfs = {}
        files = os.listdir("data") if os.path.isdir("data") else []

        for file in files:
            m = re.search(r"api_dump_(\S+).json", file)
            if not m:
                continue
            date = pd.to_datetime(m.group(1), format="%Y-%m-%d_%H-%M-%S") + pd.Timedelta(hours=1)

            with open(os.path.join("data", file), "r") as f:
                jj = json.load(f)

            for s in jj.get("data", {}).get("stations", []):
                sid = str(s.get("station_id"))

                if "vehicle_types_available" in s:
                    for v in s["vehicle_types_available"]:
                        vt = v.get("vehicle_type_id")
                        cnt = v.get("count", 0)
                        if vt is not None:
                            s[f"vehicle_count_{vt}"] = cnt

                s["date"] = date
                s.pop("vehicle_types_available", None)

                df = pd.DataFrame([s]).set_index("date")
                dfs[sid] = pd.concat([dfs.get(sid, pd.DataFrame()), df], axis=0)

        for sid in list(dfs.keys()):
            dfs[sid] = dfs[sid].sort_index()

    for sid in stations.index:
        if sid not in dfs:
            continue

        ts = dfs[sid].copy()
        for t in [192, 189, 183]:
            col = f"vehicle_count_{t}"
            if col not in ts.columns:
                ts[col] = 0
        vcols = [c for c in ts.columns if c.startswith("vehicle_count_")]
        if vcols:
            ts[vcols] = ts[vcols].fillna(0)

        if pd.isna(stations.loc[sid, "capacity"]):
            if "num_bikes_available" in ts.columns and len(ts):
                mx = pd.to_numeric(ts["num_bikes_available"], errors="coerce").max()
                if pd.notna(mx) and mx > 0:
                    stations.loc[sid, "capacity"] = round(float(mx) / 1.15)

        dfs[sid] = ts

    stations = stations.loc[[sid for sid in stations.index if sid in dfs]].copy()
    return stations, dfs


@st.cache_data(show_spinner=False)
def build_station_features(stations: pd.DataFrame, dfs: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=stations.index)
    out["name"] = stations["name"].astype(str)

    def bucket_mean(series: pd.Series, hours: set[int]) -> float:
        if series.empty:
            return 0.0
        mask = series.index.hour.isin(hours)
        s = series.loc[mask]
        if s.empty:
            return 0.0
        return float(s.mean())

    morning_hours = set(range(6, 10))
    midday_hours = set(range(10, 14))
    evening_hours = set(range(14, 18))
    late_evening_hours = set(range(18, 24))
    night_hours = set(range(0, 6))

    for sid in stations.index:
        cap_raw = stations.loc[sid, "capacity"]
        cap = float(cap_raw) if pd.notna(cap_raw) else np.nan
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0

        ts = dfs[sid]
        if "num_bikes_available" not in ts.columns or ts.empty:
            out.loc[
                sid,
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "empty_fraction",
                    "full_fraction",
                    "morning_mean",
                    "midday_mean",
                    "evening_mean",
                    "late_evening_mean",
                    "night_mean",
                ],
            ] = 0.0
            for t in [192, 189, 183]:
                out.loc[sid, f"mean-{t}"] = 0.0
            out.loc[sid, ["share-192", "share-189", "share-183", "entropy"]] = 0.0
            continue

        occ = pd.to_numeric(ts["num_bikes_available"], errors="coerce").astype(float) / cap
        occ = occ.replace([np.inf, -np.inf], np.nan).dropna()

        if occ.empty:
            out.loc[
                sid,
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "empty_fraction",
                    "full_fraction",
                    "morning_mean",
                    "midday_mean",
                    "evening_mean",
                    "late_evening_mean",
                    "night_mean",
                ],
            ] = 0.0
        else:
            out.loc[sid, "mean"] = float(occ.mean())
            out.loc[sid, "std"] = float(occ.std(ddof=0))
            out.loc[sid, "min"] = float(occ.min())
            out.loc[sid, "max"] = float(occ.max())
            out.loc[sid, "empty_fraction"] = float((occ == 0).mean())
            out.loc[sid, "full_fraction"] = float((occ >= 1).mean())

            out.loc[sid, "morning_mean"] = bucket_mean(occ, morning_hours)
            out.loc[sid, "midday_mean"] = bucket_mean(occ, midday_hours)
            out.loc[sid, "evening_mean"] = bucket_mean(occ, evening_hours)
            out.loc[sid, "late_evening_mean"] = bucket_mean(occ, late_evening_hours)
            out.loc[sid, "night_mean"] = bucket_mean(occ, night_hours)

        for t in [192, 189, 183]:
            col = f"vehicle_count_{t}"
            if col in ts.columns:
                out.loc[sid, f"mean-{t}"] = float(
                    (pd.to_numeric(ts[col], errors="coerce").fillna(0).astype(float) / cap).mean()
                )
            else:
                out.loc[sid, f"mean-{t}"] = 0.0

        total = float(out.loc[sid, ["mean-192", "mean-189", "mean-183"]].sum())
        if total > 0:
            out.loc[sid, "share-192"] = float(out.loc[sid, "mean-192"] / total)
            out.loc[sid, "share-189"] = float(out.loc[sid, "mean-189"] / total)
            out.loc[sid, "share-183"] = float(out.loc[sid, "mean-183"] / total)
        else:
            out.loc[sid, "share-192"] = 0.0
            out.loc[sid, "share-189"] = 0.0
            out.loc[sid, "share-183"] = 0.0

        p = out.loc[sid, ["share-192", "share-189", "share-183"]].to_numpy(dtype=float)
        p = np.clip(p, 0.0, 1.0)
        s = float(p.sum())
        if s > 0:
            p = p / s
        p = p[p > 0]
        out.loc[sid, "entropy"] = float(-np.sum(p * np.log(p))) if p.size else 0.0

    num_cols = [c for c in out.columns if c != "name"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def cluster_stations(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    NAME_COL = "name"
    feats = df.copy()
    feature_cols = [c for c in feats.columns if c != NAME_COL]
    feats[feature_cols] = feats[feature_cols].apply(pd.to_numeric, errors="coerce")
    feats[feature_cols] = feats[feature_cols].replace([np.inf, -np.inf], np.nan)
    feats[feature_cols] = feats[feature_cols].fillna(feats[feature_cols].median(numeric_only=True))

    if feats[feature_cols].empty:
        out = df.copy()
        out["cluster"] = 0
        return out

    X = feats[feature_cols].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    raw_labels = km.fit_predict(X_scaled)

    out = df.copy()
    out["cluster_raw"] = raw_labels

    if "mean" in out.columns:
        cm = out.groupby("cluster_raw")["mean"].mean().sort_values()
        ordered = list(cm.index)
        if len(ordered) == 3:
            remap = {ordered[0]: 2, ordered[1]: 0, ordered[2]: 1}
        else:
            remap = {c: i for i, c in enumerate(ordered)}
        out["cluster"] = out["cluster_raw"].map(remap).astype(int)
    else:
        out["cluster"] = out["cluster_raw"].astype(int)

    out = out.drop(columns=["cluster_raw"])
    return out


@st.cache_data(show_spinner=False)
def build_long_df(stations: pd.DataFrame, dfs: dict) -> pd.DataFrame:
    rows = []
    for sid in stations.index:
        cap_raw = stations.loc[sid, "capacity"]
        cap = float(cap_raw) if pd.notna(cap_raw) else np.nan
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0

        ts = dfs[sid].reset_index().copy()
        if ts.empty:
            continue
        if "date" not in ts.columns:
            ts = ts.rename(columns={ts.columns[0]: "date"})
        ts["station_id"] = str(sid)
        if "num_bikes_available" in ts.columns:
            ts["occupancy_rate"] = pd.to_numeric(ts["num_bikes_available"], errors="coerce").astype(float) / cap
        else:
            ts["occupancy_rate"] = np.nan

        for t in [192, 189, 183]:
            col = f"vehicle_count_{t}"
            if col not in ts.columns:
                ts[col] = 0.0

        rows.append(
            ts[
                [
                    "date",
                    "station_id",
                    "occupancy_rate",
                    "vehicle_count_192",
                    "vehicle_count_189",
                    "vehicle_count_183",
                ]
            ]
        )

    if not rows:
        out = pd.DataFrame(
            columns=[
                "date",
                "station_id",
                "occupancy_rate",
                "vehicle_count_192",
                "vehicle_count_189",
                "vehicle_count_183",
            ]
        )
        out["date"] = pd.to_datetime(out["date"])
        return out

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.sort_values("date")


cluster_order = [2, 0, 1]
cluster_colors = {2: "#d62728", 0: "#1f77b4", 1: "#006400"}

cluster_meta = {
    2: {
        "title": "Low Availability Station",
        "desc": "Low average occupancy with frequent emptiness, almost never full<br>(mean occupancy rate = 0.232, std = 0.157, empty = 11.6%, full = 1.1%)",
    },
    0: {
        "title": "Balanced Station",
        "desc": "Moderate occupancy with variability; rarely empty and only occasionally full<br>(mean occupancy rate = 0.711, std = 0.211, empty = 0.7%, full = 16.7%)",
    },
    1: {
        "title": "Persistently Saturated Central Hub",
        "desc": "Consistently high occupancy; almost never empty but very frequently full<br>(mean occupancy rate = 1.117, std = 0.226, empty = 0.05%, full = 69.6%)",
    },
}


def marker_area_from_capacity(cap):
    return float(np.clip(20 + 2 * cap, 20, 80))


def radius_from_capacity(cap):
    area = marker_area_from_capacity(cap)
    return float(np.sqrt(area) * 0.55)


def build_base_folium_map(map_df: pd.DataFrame):
    center_lat = float(map_df["lat"].mean()) if len(map_df) else 48.2082
    center_lon = float(map_df["lon"].mean()) if len(map_df) else 16.3738

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="CartoDB Positron",
        control=False,
    ).add_to(m)

    rows = map_df[["station_id", "name", "lat", "lon", "capacity", "cluster"]].to_dict("records")

    for cl in cluster_order:
        for r in rows:
            if int(r["cluster"]) != int(cl):
                continue
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=radius_from_capacity(float(r["capacity"])),
                color=cluster_colors.get(int(cl), "#444444"),
                weight=0,
                fill=True,
                fill_color=cluster_colors.get(int(cl), "#444444"),
                fill_opacity=0.75,
                tooltip=f"{r['name']}",
            ).add_to(m)

    cap_min = int(map_df["capacity"].min()) if len(map_df) else 0
    cap_max = int(map_df["capacity"].max()) if len(map_df) else 0

    usage_items = ""
    for cl in cluster_order:
        meta = cluster_meta.get(cl, {"title": f"Cluster {cl}", "desc": ""})
        usage_items += f"""
        <div style=\"margin-bottom:10px;\">
          <div style=\"display:flex; align-items:center; gap:8px;\">
            <span style=\"width:10px;height:10px;border-radius:50%;background:{cluster_colors[cl]};display:inline-block;\"></span>
            <span style=\"font-weight:700;\">{meta['title']}</span>
          </div>
          <div style=\"margin-left:18px; font-size:12px; line-height:1.25; color:#333;\">{meta['desc']}</div>
        </div>
        """

    legend_html = f"""
    <div id=\"legend-box\" style=\"
      position: fixed;
      left: 20px;
      bottom: 20px;
      z-index: 9999;
      background: rgba(255,255,255,0.95);
      padding: 0;
      border-radius: 10px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      width: 420px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      overflow: hidden;
    \" >
      <div id=\"legend-header\" style=\"
        padding: 10px 14px;
        cursor: pointer;
        user-select: none;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:10px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
      \" onclick=\"(function(){{var b=document.getElementById('legend-body'); if(b.style.display==='none'){{b.style.display='block';}} else {{b.style.display='none';}}}})()\">
        <div style=\"font-size:16px; font-weight:800; line-height:1.1;\">Legend</div>
        <div style=\"font-size:18px; color:#333;\">▾</div>
      </div>

      <div id=\"legend-body\" style=\"padding: 12px 14px; display:none;\">
        <div style=\"font-size:12px; color:#444; margin-bottom:10px;\">Colored by usage type, scaled by capacity</div>

        <div style=\"font-size:14px; font-weight:800; margin-bottom:8px;\">Usage Type</div>
        {usage_items}

        <div style=\"height:8px;\"></div>
        <div style=\"font-size:14px; font-weight:800; margin-bottom:8px;\">Capacity</div>
        <div style=\"font-size:12px; color:#333;\">
          Smallest: <b>{cap_min}</b> bikes &nbsp;|&nbsp; Largest: <b>{cap_max}</b> bikes
        </div>

        <div style=\"height:8px;\"></div>
        <div style=\"font-size:11px; color:#666;\">Click a station on the map to update the panels.</div>
      </div>
    </div>
    """

    class Legend(MacroElement):
        def __init__(self, html):
            super().__init__()
            self._template = Template(f"{{% macro html(this, kwargs) %}}{html}{{% endmacro %}}")

    m.get_root().add_child(Legend(legend_html))

    return m


def build_folium_map(map_df: pd.DataFrame):
    if "_base_map" not in st.session_state:
        st.session_state._base_map = build_base_folium_map(map_df)
    return st.session_state._base_map


def select_station_from_click(map_df: pd.DataFrame, click_lat: float, click_lon: float) -> str:
    tmp = map_df[["station_id", "lat", "lon"]].copy()
    tmp["dist2"] = (tmp["lat"] - float(click_lat)) ** 2 + (tmp["lon"] - float(click_lon)) ** 2
    return str(tmp.sort_values("dist2").iloc[0]["station_id"])


if "selected_station" not in st.session_state:
    st.session_state.selected_station = None
if "_map_nonce" not in st.session_state:
    st.session_state._map_nonce = 0


stations, dfs = load_data()
features = build_station_features(stations, dfs)
features = cluster_stations(features, k=3)
ts_long = build_long_df(stations, dfs)

map_df = stations.join(features[["cluster"]], how="inner").reset_index()
map_df["station_id"] = map_df["station_id"].astype(str)
map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
map_df["capacity"] = pd.to_numeric(map_df["capacity"], errors="coerce")
map_df["cluster"] = pd.to_numeric(map_df["cluster"], errors="coerce")
map_df = map_df.dropna(subset=["lat", "lon", "capacity", "cluster"])
map_df["cluster"] = map_df["cluster"].astype(int)

sum_stats = features[["name", "mean", "std"]].copy()


st.title("WienMobil Bicycle Stations – Explorer")

col_map, col_details = st.columns([2.2, 1.0], gap="large")

with col_map:
    st.subheader("Bike Stations on the Map")

    if map_df.empty:
        st.error("Map dataframe is empty after cleaning lat/lon/capacity/cluster.")
    else:
        m = build_folium_map(map_df)
        out = st_folium(m, height=520, use_container_width=True, key=f"map_{st.session_state.get('_map_nonce', 0)}")

        if out and out.get("last_clicked"):
            sid = select_station_from_click(map_df, out["last_clicked"]["lat"], out["last_clicked"]["lng"])
            st.session_state.selected_station = sid

with col_details:
    st.subheader("Station details")
    sid = st.session_state.selected_station

    if sid is None:
        st.info("No station selected – click on a station in the map.")
    else:
        if sid in stations.index:
            s = stations.loc[sid]
            f = features.loc[sid]

            station_name = str(s.get("name", "(no name)"))
            cap_val = int(float(s.get("capacity", np.nan))) if pd.notna(s.get("capacity", np.nan)) else "-"
            cluster_title = cluster_meta.get(int(f["cluster"]), {"title": f"Cluster {int(f['cluster'])}"})["title"]

            st.markdown(
                f"""
                <div style='font-size:12px;color:#666;font-weight:600;letter-spacing:0.02em;'>Station name</div>
                <div style='font-size:22px;font-weight:700;line-height:1.2;margin-bottom:14px;'>{station_name}</div>

                <div style='font-size:12px;color:#666;font-weight:600;letter-spacing:0.02em;'>Capacity</div>
                <div style='font-size:22px;font-weight:700;line-height:1.2;margin-bottom:14px;'>{cap_val}</div>

                <div style='font-size:12px;color:#666;font-weight:600;letter-spacing:0.02em;'>Usage Type</div>
                <div style='font-size:22px;font-weight:700;line-height:1.2;margin-bottom:10px;'>{cluster_title}</div>
                """,
                unsafe_allow_html=True,
            )

            show_cols = [
                "mean",
                "std",
                "min",
                "max",
                "empty_fraction",
                "full_fraction",
            ]
            col_names = [
                "Mean Occupancy Rate (OccR)",
                "Standard Deviation of OccR",
                "Minimum Recorded OccR",
                "Maximum Recorded OccR",
                "Fraction of the time being empty",
                "Fraction of the time being full",
            ]
            st.markdown(
                "<div style='font-size:12px;color:#666;font-weight:600;letter-spacing:0.02em;margin-top:6px;'>Metrics</div>",
                unsafe_allow_html=True,
            )
            df_show = pd.DataFrame({"metric": col_names, "value": [f"{round(float(f[c]) * 100)}%" for c in show_cols]})
            st.dataframe(df_show, hide_index=True, use_container_width=True)
        else:
            st.warning("Selected station not found in stations index.")

    spacer_cols = st.columns([1, 1, 1])
    with spacer_cols[1]:
        if st.button("Reset selection", use_container_width=True):
            st.session_state.selected_station = None
            st.session_state._map_nonce = int(st.session_state.get("_map_nonce", 0)) + 1
            st.rerun()


sid = st.session_state.selected_station
if sid is None:
    st.subheader("Average occupancy rates of all stations for one week")
else:
    station_name = stations.loc[sid, "name"] if sid in stations.index else str(sid)
    st.subheader(f"Occupancy rates of \"{station_name}\" for one week")

if sid is None:
    if ts_long.empty:
        st.info("No time series data available.")
    else:
        agg = ts_long.groupby("date", as_index=False)["occupancy_rate"].mean()
        fig_ts = px.line(agg, x="date", y="occupancy_rate", height=360)
        fig_ts.update_traces(line_color="#1f77b4")
        fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Occupancy rate", xaxis_title="")
        fig_ts.update_xaxes(
            tickformat="%d.%m<br>(%a)",
            tickformatstops=[
                dict(dtickrange=[None, 12 * 60 * 60 * 1000], value="%H:%M"),
                dict(dtickrange=[12 * 60 * 60 * 1000, 36 * 60 * 60 * 1000], value="%d.%m<br>%H:%M"),
                dict(dtickrange=[36 * 60 * 60 * 1000, None], value="%d.%m<br>(%a)"),
            ],
        )
        st.plotly_chart(fig_ts, use_container_width=True)
else:
    d = ts_long[ts_long["station_id"] == sid]
    if d.empty:
        st.info("No time series data for selected station.")
    else:
        fig_ts = px.line(d, x="date", y="occupancy_rate", height=360)
        fig_ts.update_traces(line_color="#1f77b4")
        fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Occupancy rate", xaxis_title="")
        fig_ts.update_xaxes(
            tickformat="%d.%m<br>(%a)",
            tickformatstops=[
                dict(dtickrange=[None, 12 * 60 * 60 * 1000], value="%H:%M"),
                dict(dtickrange=[12 * 60 * 60 * 1000, 36 * 60 * 60 * 1000], value="%d.%m<br>%H:%M"),
                dict(dtickrange=[36 * 60 * 60 * 1000, None], value="%d.%m<br>(%a)"),
            ],
        )
        st.plotly_chart(fig_ts, use_container_width=True)


col_p3, col_p4 = st.columns([1.25, 1.0], gap="large")

with col_p3:
    sid = st.session_state.selected_station
    if sid is None:
        st.subheader("Average occupancy rates of all stations")
    else:
        station_name = stations.loc[sid, "name"] if sid in stations.index else str(sid)
        st.subheader(f"Average occupancy rates distribution – \"{station_name}\"")

    means = pd.to_numeric(sum_stats["mean"], errors="coerce").dropna().astype(float)
    if means.empty:
        st.info("No station mean occupancy data available.")
    else:
        nbins = 40
        hist_counts, bin_edges = np.histogram(means.to_numpy(), bins=nbins)
        bin_left = bin_edges[:-1]
        bin_right = bin_edges[1:]
        bin_center = (bin_left + bin_right) / 2.0
        bin_width = (bin_right - bin_left)

        selected_sid = st.session_state.selected_station
        selected_mean = None
        selected_name = None

        if selected_sid is not None and str(selected_sid) in sum_stats.index.astype(str).tolist():
            selected_sid = str(selected_sid)
            selected_mean = float(sum_stats.loc[selected_sid, "mean"])
            selected_name = str(sum_stats.loc[selected_sid, "name"]) if "name" in sum_stats.columns else selected_sid

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Bar(
                x=bin_center,
                y=hist_counts,
                width=bin_width,
                marker=dict(color="#1f77b4", opacity=0.85),
                customdata=np.stack([bin_left, bin_right, hist_counts], axis=1),
                hovertemplate=(
                    "Range: [%{customdata[0]:.2f}, %{customdata[1]:.2f})<br>"
                    "Count: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        if selected_mean is not None:
            fig_hist.add_vline(x=selected_mean, line_dash="dash", line_color="red")
            fig_hist.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name=f"{selected_name} ({selected_mean:.2f})",
                    showlegend=True,
                )
            )

        fig_hist.add_vline(x=1.0, line_color="black")
        fig_hist.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="black"),
                name="fully occupied (1.0)",
                showlegend=True,
            )
        )

        fig_hist.update_layout(
            xaxis_title="Mean occupancy rate",
            yaxis_title="Number of stations",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )

        st.plotly_chart(fig_hist, use_container_width=True)

with col_p4:
    sid = st.session_state.selected_station
    if sid is None:
        st.subheader("Bike Types Available in all of Vienna")
    else:
        station_name = stations.loc[sid, "name"] if sid in stations.index else str(sid)
        st.subheader(f"Bike Types Available at \"{station_name}\"")

    if sid is None:
        if ts_long.empty:
            m = pd.Series({"vehicle_count_192": 0.0, "vehicle_count_189": 0.0, "vehicle_count_183": 0.0})
        else:
            m = ts_long[["vehicle_count_192", "vehicle_count_189", "vehicle_count_183"]].mean(numeric_only=True)
    else:
        d = ts_long[ts_long["station_id"] == sid]
        if d.empty:
            m = pd.Series({"vehicle_count_192": 0.0, "vehicle_count_189": 0.0, "vehicle_count_183": 0.0})
        else:
            m = d[["vehicle_count_192", "vehicle_count_189", "vehicle_count_183"]].mean(numeric_only=True)

    total = float(m.sum())
    shares = (m / total) if total > 0 else m * 0.0

    dist = pd.DataFrame(
        {
            "Bike type": ["Standard", "Cargo-Bike", "E-Bike"],
            "Share (%)": [
                100 * float(shares.get("vehicle_count_192", 0.0)),
                100 * float(shares.get("vehicle_count_189", 0.0)),
                100 * float(shares.get("vehicle_count_183", 0.0)),
            ],
        }
    )

    fig_bar = px.bar(
        dist,
        x="Bike type",
        y="Share (%)",
        text=dist["Share (%)"].map(lambda x: f"{x:.1f}%"),
        height=420,
    )
    fig_bar.update_traces(textposition="outside", marker_color="#1f77b4")
    fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Share (%)", xaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)
