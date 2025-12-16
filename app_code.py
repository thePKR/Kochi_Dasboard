#Spelling mistake is intentional as it sounds cool . Anyways here are the # app.py
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import os
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Kochi Transport Dashboard", page_icon="🚏")

# -----------------------
# Config - file names (place files in same folder)
# -----------------------
SHAPEFILE = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\KMCShapeFile"                   # or GeoJSON like kochi_wards.geojson
BUS_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\Bus_stops Cleaned.csv"
AUTO_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\auto_cleaned.csv"
METRO_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\Metro_cleaned.csv"
ROAD_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\Road_cleaned.csv"
TRANS_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\CLEANED_Transportation_Best_Way_to_Reach.csv"
INDICATORS_CSV = r"D:\\Study\\Fieldwork\\Year 2\\transport_app_4\\wards_transport_indicators.csv" 

# Palette: RED (lowest) -> GREEN (middle) -> BLUE (highest)
PALETTE = ['#FF0000', '#00FF00', '#0000FF']
NO_DATA_COLOR = "#808080"  # grey for wards with no data

# -----------------------
# Helpers
# -----------------------
def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def normalize_ward_col(df):
    for c in df.columns:
        if 'ward' in c.lower():
            df = df.copy()
            df['ward'] = df[c].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            return df
    df = df.copy()
    df['ward'] = df.index.astype(str)
    return df

def detect_length_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ['length','len','km','m','meter','dist','distance']):
            return c
    return None

def detect_seat_cols(df):
    return [c for c in df.columns if any(k in c.lower() for k in ['seat','capacity','avail','percent'])]

def normalize_series(s):
    s = s.astype(float)
    if s.isnull().all():
        return s.fillna(0)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return s.fillna(0)
    return (s - mn) / (mx - mn)

# -----------------------
# Load shapefile robustly (attributes and geometry if available)
# -----------------------
try:
    shp = gpd.read_file(SHAPEFILE)
except Exception:
    # fallback attributes-only
    shp = gpd.read_file(SHAPEFILE, ignore_geometry=True)

# Normalize ward field in shapefile attributes
if 'ward' not in shp.columns:
    if 'sourcewa_1' in shp.columns:
        shp['ward'] = shp['sourcewa_1'].astype(str)
    elif 'ward_lgd_n' in shp.columns:
        shp['ward'] = shp['ward_lgd_n'].astype(str)
    elif 'WARD' in shp.columns:
        shp['ward'] = shp['WARD'].astype(str)
    else:
        shp['ward'] = shp.index.astype(str)

# -----------------------
# Load CSVs & compute indicators (as before)
# -----------------------
bus_df = normalize_ward_col(safe_read_csv(BUS_CSV))
auto_df = normalize_ward_col(safe_read_csv(AUTO_CSV))
metro_df = normalize_ward_col(safe_read_csv(METRO_CSV))
road_df = normalize_ward_col(safe_read_csv(ROAD_CSV))
trans_df = normalize_ward_col(safe_read_csv(TRANS_CSV))

wards = shp[['ward']].copy()
if 'st_area(sh' in shp.columns:
    wards['area_km2'] = (shp['st_area(sh'] / 1e6).round(4)
else:
    wards['area_km2'] = np.nan

wards = wards.merge(bus_df.groupby('ward').size().rename('bus_count').reset_index(), on='ward', how='left')
wards = wards.merge(auto_df.groupby('ward').size().rename('auto_count').reset_index(), on='ward', how='left')
wards = wards.merge(metro_df.groupby('ward').size().rename('metro_count').reset_index(), on='ward', how='left')
wards['bus_count'] = wards['bus_count'].fillna(0).astype(int)
wards['auto_count'] = wards['auto_count'].fillna(0).astype(int)
wards['metro_count'] = wards['metro_count'].fillna(0).astype(int)

# taxi detection
taxi_col = None
for df in [auto_df, road_df, bus_df, trans_df]:
    for c in df.columns:
        if 'taxi' in c.lower() and any(k in c.lower() for k in ['count','num','no','number']):
            taxi_col = c
            break
    if taxi_col:
        break

if taxi_col:
    parts = []
    for df in [auto_df, road_df, bus_df, trans_df]:
        if taxi_col in df.columns and 'ward' in df.columns:
            parts.append(df[['ward', taxi_col]])
    if parts:
        taxi_all = pd.concat(parts, ignore_index=True, sort=False)
        taxi_all[taxi_col] = pd.to_numeric(taxi_all[taxi_col], errors='coerce')
        taxi_agg = taxi_all.groupby('ward')[taxi_col].sum().reset_index().rename(columns={taxi_col:'taxi_count'})
        wards = wards.merge(taxi_agg, on='ward', how='left')

if 'taxi_count' not in wards.columns:
    wards['taxi_count'] = 0
else:
    wards['taxi_count'] = wards['taxi_count'].fillna(0).astype(int)

# distances & seats
dist_cols = [c for c in trans_df.columns if any(k in c.lower() for k in ['airport','ernakul','ernak','north','south','rail','dist','distance','km','m'])]
seat_cols = detect_seat_cols(trans_df)
for c in dist_cols + seat_cols:
    trans_df[c] = pd.to_numeric(trans_df[c], errors='coerce')
if not trans_df.empty and 'ward' in trans_df.columns and (dist_cols or seat_cols):
    trans_agg = trans_df.groupby('ward')[dist_cols + seat_cols].mean().reset_index()
    wards = wards.merge(trans_agg, on='ward', how='left')

# population estimate
pop_cols = [c for df in [bus_df,auto_df,metro_df,road_df,trans_df] for c in df.columns if 'pop' in c.lower()]
if pop_cols:
    pcol = pop_cols[0]
    if '2025' in pcol:
        wards['population'] = wards.get(pcol, np.nan).astype('Int64')
    else:
        wards['population'] = (wards.get(pcol, 0).fillna(0) * 1.458).round().astype(int)
else:
    TOTAL_2025 = 924000
    if wards['area_km2'].isnull().any():
        wards['population'] = int(TOTAL_2025 / len(wards))
    else:
        wards['area_share'] = wards['area_km2'] / wards['area_km2'].sum()
        wards['population'] = (wards['area_share'] * TOTAL_2025).round().astype(int)

# per-1k indicators
wards['buses_per_1000'] = (wards['bus_count'] / (wards['population'] / 1000)).round(3)
wards['autos_per_1000'] = (wards['auto_count'] / (wards['population'] / 1000)).round(3)
wards['taxis_per_1000'] = (wards['taxi_count'] / (wards['population'] / 1000)).round(3)
wards['metros_per_1000'] = (wards['metro_count'] / (wards['population'] / 1000)).round(3)

# road length
road_len_col = detect_length_col(road_df)
if road_len_col and 'ward' in road_df.columns:
    road_df[road_len_col+'_num'] = pd.to_numeric(road_df[road_len_col], errors='coerce')
    med = road_df[road_len_col+'_num'].median(skipna=True)
    if pd.notna(med) and med > 100:
        road_df['road_km'] = road_df[road_len_col+'_num'] / 1000.0
    else:
        road_df['road_km'] = road_df[road_len_col+'_num']
    road_agg = road_df.groupby('ward', as_index=False)['road_km'].sum().rename(columns={'road_km':'road_km_total'})
    wards = wards.merge(road_agg, on='ward', how='left')

wards['road_km_total'] = wards.get('road_km_total', 0.0).fillna(0.0)
wards['road_km_per_1000'] = (wards['road_km_total'] / (wards['population'] / 1000)).round(3)

# intermodal score default
wards['bus_n'] = normalize_series(wards['buses_per_1000'].fillna(0))
wards['auto_n'] = normalize_series(wards['autos_per_1000'].fillna(0))
wards['taxi_n'] = normalize_series(wards['taxis_per_1000'].fillna(0))
wards['metro_n'] = normalize_series(wards['metros_per_1000'].fillna(0))
inv_col = None
for c in dist_cols:
    if any(k in c.lower() for k in ['airport','north','south']):
        inv_col = c
        break
if inv_col:
    inv_val = 1.0 / wards[inv_col].replace(0, np.nan)
    wards['inv_n'] = normalize_series(inv_val.fillna(0))
else:
    wards['inv_n'] = 0
wards['intermodal_score'] = (wards['bus_n']*0.4 + wards['auto_n']*0.2 + wards['taxi_n']*0.1 + wards['metro_n']*0.2 + wards['inv_n']*0.1) * 100

# save indicators
try:
    wards.to_csv(INDICATORS_CSV, index=False)
except Exception:
    pass

# friendly labels
LABELS = {
    'buses_per_1000': "Bus stops per 1,000 people",
    'autos_per_1000': "Auto stands per 1,000 people",
    'taxis_per_1000': "Taxis per 1,000 people",
    'metros_per_1000': "Metro stations per 1,000 people",
    'population': "Estimated population (2025)",
    'road_km_per_1000': "Road length (km) per 1,000 people",
    'intermodal_score': "Transport connectivity score (0–100)"
}

# interactive weights
st.sidebar.header("Set what matters most")
bus_w = st.sidebar.slider("Bus importance", 0.0, 1.0, 0.4, 0.05)
metro_w = st.sidebar.slider("Metro importance", 0.0, 1.0, 0.2, 0.05)
auto_w = st.sidebar.slider("Auto importance", 0.0, 1.0, 0.2, 0.05)
taxi_w = st.sidebar.slider("Taxi importance", 0.0, 1.0, 0.1, 0.05)
inv_w = st.sidebar.slider("Closeness importance", 0.0, 1.0, 0.1, 0.05)
weights = {'bus': bus_w, 'metro': metro_w, 'auto': auto_w, 'taxi': taxi_w, 'invdist': inv_w}

def compute_custom(df, weights):
    d = df.copy()
    d['bus_n'] = normalize_series(d['buses_per_1000'].fillna(0))
    d['metro_n'] = normalize_series(d['metros_per_1000'].fillna(0))
    d['auto_n'] = normalize_series(d['autos_per_1000'].fillna(0))
    d['taxi_n'] = normalize_series(d['taxis_per_1000'].fillna(0))
    if inv_col:
        invv = 1.0 / d[inv_col].replace(0, np.nan)
        d['inv_n'] = normalize_series(invv.fillna(0))
    else:
        d['inv_n'] = 0
    tot = sum(weights.values()) or 1.0
    d['custom_score'] = (d['bus_n']*weights['bus'] + d['metro_n']*weights['metro'] + d['auto_n']*weights['auto'] + d['taxi_n']*weights['taxi'] + d['inv_n']*weights['invdist'])/tot * 100
    return d

wards_custom = compute_custom(wards, weights)

# UI header & KPIs
st.title("Kochi — Ward transport overview (plain labels)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wards", f"{len(wards_custom)}")
c2.metric(LABELS['population'], f"{int(wards_custom['population'].sum()):,}")
c3.metric(LABELS['intermodal_score'], f"{wards_custom['intermodal_score'].mean():.2f}")
c4.metric(LABELS['road_km_per_1000'], f"{wards_custom['road_km_per_1000'].mean():.2f}")
st.markdown("---")

# Map selection
st.subheader("Map — choose what to display")
map_choice = st.selectbox("Map shows", options=[
    ('intermodal_score', LABELS['intermodal_score']),
    ('custom_score', "Personalized connectivity score"),
    ('buses_per_1000', LABELS['buses_per_1000']),
    ('autos_per_1000', LABELS['autos_per_1000']),
    ('taxis_per_1000', LABELS['taxis_per_1000']),
    ('metros_per_1000', LABELS['metros_per_1000']),
    ('road_km_per_1000', LABELS['road_km_per_1000']),
    ('population', LABELS['population'])
], format_func=lambda x: x[1])[0]

value_col = map_choice

# Try to read geometry and merge
gdf_geo = None
try:
    gdf_geo = gpd.read_file(SHAPEFILE)
    if 'ward' not in gdf_geo.columns:
        if 'sourcewa_1' in gdf_geo.columns:
            gdf_geo['ward'] = gdf_geo['sourcewa_1'].astype(str)
        elif 'ward_lgd_n' in gdf_geo.columns:
            gdf_geo['ward'] = gdf_geo['ward_lgd_n'].astype(str)
        elif 'WARD' in gdf_geo.columns:
            gdf_geo['ward'] = gdf_geo['WARD'].astype(str)
        else:
            gdf_geo['ward'] = gdf_geo.index.astype(str)
    # merge attributes
    gdf_geo = gdf_geo.merge(wards_custom, on='ward', how='left')
    # to lat/lon for folium
    try:
        gdf_geo = gdf_geo.to_crs(epsg=4326)
    except Exception:
        pass
except Exception:
    gdf_geo = None

# Map rendering with grey no-data and bottom-left legend
if gdf_geo is not None and 'geometry' in gdf_geo.columns:
    m = folium.Map(location=[9.9667, 76.2894], zoom_start=11, tiles="cartodbpositron")
    vals = gdf_geo[value_col].replace([np.inf,-np.inf], np.nan)
    vmin = float(vals.min(skipna=True)) if not vals.dropna().empty else 0.0
    vmax = float(vals.max(skipna=True)) if not vals.dropna().empty else 1.0

    # style function: grey if NaN
    def style_fn(feature):
        wardid = str(feature['properties'].get('ward'))
        row = gdf_geo[gdf_geo['ward'].astype(str) == wardid]
        if row.empty or pd.isna(row.iloc[0][value_col]):
            return {'fillColor': NO_DATA_COLOR, 'color': '#444444', 'weight':0.3, 'fillOpacity':0.85}
        val = row.iloc[0][value_col]
        # compute color from red->green->blue gradient
        # normalized t in [0,1]
        if vmax==vmin:
            t = 0.5
        else:
            t = (val - vmin) / (vmax - vmin)
        # map t through piecewise gradient:
        # 0 -> red (#FF0000), 0.5 -> green (#00FF00), 1 -> blue (#0000FF)
        if t <= 0.5:
            # red -> green
            r = int(255 * (1 - 2*t))
            g = int(255 * (2*t))
            b = 0
        else:
            # green -> blue
            t2 = (t - 0.5) * 2
            r = 0
            g = int(255 * (1 - t2))
            b = int(255 * t2)
        color = f'#{r:02x}{g:02x}{b:02x}'
        return {'fillColor': color, 'color':'#444444', 'weight':0.3, 'fillOpacity':0.85}

    folium.GeoJson(
        gdf_geo.to_json(),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=['ward', value_col], aliases=['Ward', LABELS.get(value_col, value_col)], localize=True)
    ).add_to(m)

    # custom bottom-left legend (HTML)
    # gradient bar CSS from red->green->blue
    legend_html = f'''
     <div style="
         position: fixed;
         bottom: 30px;
         left: 10px;
         z-index:9999;
         background:white;
         padding:8px 10px;
         border-radius:6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.15);
         font-size:12px;
     ">
       <div style="font-weight:700; margin-bottom:6px;">{LABELS.get(value_col, value_col)}</div>
       <div style="width:220px; height:12px; background: linear-gradient(to right, #FF0000, #00FF00, #0000FF); border:1px solid #ddd;"></div>
       <div style="display:flex; justify-content:space-between; margin-top:4px;">
         <span style="font-size:11px;">{vmin:.2f}</span>
         <span style="font-size:11px;">{vmax:.2f}</span>
       </div>
       <div style="text-align:center; margin-top:6px; font-size:11px; color:#555;">Red = Low · Green = Medium · Blue = High · Grey = No data</div>
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    st_data = st_folium(m, width=1000, height=600)
else:
    st.info("Polygons not available here — showing colored preview bar chart.")
    preview = wards_custom.sort_values(value_col, ascending=True).reset_index(drop=True)
    # generate colors same way as map (red->green->blue; grey for NaN)
    vmin = float(preview[value_col].min(skipna=True)) if not preview[value_col].dropna().empty else 0.0
    vmax = float(preview[value_col].max(skipna=True)) if not preview[value_col].dropna().empty else 1.0
    def value_color(v):
        if pd.isna(v): return NO_DATA_COLOR
        if vmax==vmin:
            t = 0.5
        else:
            t = (v - vmin) / (vmax - vmin)
        if t <= 0.5:
            r = int(255 * (1 - 2*t))
            g = int(255 * (2*t))
            b = 0
        else:
            t2 = (t - 0.5) * 2
            r = 0
            g = int(255 * (1 - t2))
            b = int(255 * t2)
        return f'#{r:02x}{g:02x}{b:02x}'
    colors = [value_color(v) for v in preview[value_col]]
    fig, ax = plt.subplots(figsize=(12,10))
    ax.barh(preview['ward'], preview[value_col], color=colors)
    ax.set_xlabel(LABELS.get(value_col, value_col))
    ax.set_title(f"Preview: {LABELS.get(value_col, value_col)} (Grey = no data)")
    plt.tight_layout()
    st.pyplot(fig)

# Ranking and table remain unchanged (friendly labels), omitted here for brevity
st.subheader("Top wards by selected metric")
rank_metric = st.selectbox("Choose ranking metric", options=[
    ('custom_score', "Personalized score (based on sliders)"),
    ('intermodal_score', LABELS['intermodal_score']),
    ('buses_per_1000', LABELS['buses_per_1000']),
    ('autos_per_1000', LABELS['autos_per_1000']),
    ('metros_per_1000', LABELS['metros_per_1000']),
    ('road_km_per_1000', LABELS['road_km_per_1000'])
], format_func=lambda x: x[1])[0]

source = wards_custom if rank_metric == 'custom_score' else wards
rank_df = source[['ward', rank_metric]].dropna().sort_values(rank_metric, ascending=False)
top_n = st.slider("Number of top wards", 3, 20, 8)
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(rank_df['ward'].head(top_n), rank_df[rank_metric].head(top_n))
ax2.set_xlabel("Ward")
ax2.set_ylabel(LABELS.get(rank_metric, rank_metric))
ax2.set_title(f"Top {top_n} wards by {LABELS.get(rank_metric, rank_metric)}")
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("Ward table & download")
display_df = wards_custom.rename(columns={
    'ward':'Ward',
    'population':'Estimated population (2025)',
    'buses_per_1000': LABELS['buses_per_1000'],
    'autos_per_1000': LABELS['autos_per_1000'],
    'taxis_per_1000': LABELS['taxis_per_1000'],
    'metros_per_1000': LABELS['metros_per_1000'],
    'road_km_per_1000': LABELS['road_km_per_1000'],
    'intermodal_score': LABELS['intermodal_score'],
}).fillna(0)
cols_show = ['Ward','Estimated population (2025)', LABELS['buses_per_1000'], LABELS['autos_per_1000'], LABELS['taxis_per_1000'], LABELS['metros_per_1000'], LABELS['road_km_per_1000'], LABELS['intermodal_score'], 'custom_score']
available_cols = [c for c in cols_show if c in display_df.columns]
st.dataframe(display_df[available_cols].sort_values('Ward').reset_index(drop=True))
csv_bytes = display_df[available_cols].to_csv(index=False).encode('utf-8')
st.download_button("Download data (CSV)", data=csv_bytes, file_name="wards_plain_labels.csv", mime="text/csv")

# "How these numbers are calculated" section retained
st.markdown("---")
st.header("How these numbers are calculated (simple language)")
st.markdown("""
- **Bus stops per 1,000 people:** Count of bus stops in the ward divided by ward population (per 1,000).  
- **Auto stands per 1,000 people:** Count of auto stands in the ward divided by ward population (per 1,000).  
- **Taxis per 1,000 people:** Count of taxi stands in the ward divided by ward population (per 1,000).  
- **Metro stations per 1,000 people:** Count of metro stations in the ward divided by ward population (per 1,000).  
- **Road length (km) per 1,000 people:** Sum of road lengths in the ward (km) divided by ward population (per 1,000).  
- **Transport connectivity score (0–100):** A combined score that weighs availability of modes and closeness to major hubs. You can change the weights using sliders.
""")
