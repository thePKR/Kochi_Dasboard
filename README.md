<div align="center">

# 🚏 Kochi Urban Transport Intelligence System
### *Ward-Level Analytics • Intermodal Index Engine • Spatial Policy Dashboard*

<br>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/GeoPandas-Spatial%20Analytics-2E8B57" />
<img src="https://img.shields.io/badge/Folium-Geo%20Visualization-3CB371" />
<img src="https://img.shields.io/badge/SDG-11%20Sustainable%20Cities-2E8B57" />
<img src="https://img.shields.io/badge/License-MIT-black" />

<br><br>

**From Field Immersion to Decision Intelligence**  
A policy-oriented transport analytics engine built on real ward-level data from Kochi.

</div>

---

## 🌍 Overview

This project transforms raw urban transport data into a **ward-level decision-support system**.

Instead of visualizing counts, it computes:

- Infrastructure intensity per 1,000 population  
- Multi-modal balance indicators  
- A customizable Intermodal Connectivity Index  
- Spatial equity diagnostics  

This is not a static dashboard.  
It is a **transport policy simulation tool.**

---

# 🧠 Intermodal Connectivity Engine

### Composite Index (0–100)

```
Index = w₁(Bus) + w₂(Metro) + w₃(Auto) + w₄(Taxi) + w₅(Inverse Distance)
```

✔ Min-max normalized  
✔ Weight-adjustable in real time  
✔ Live ward rank recalculation  
✔ Capital allocation experimentation ready  

Change weights → Observe structural bias → Identify under-served wards.

---

# 🗺 Spatial Intelligence Layer

- Ward shapefile integration  
- Interactive Folium choropleth  
- Auto-zoom on selected wards  
- Sustainability gradient (Purple → Blue → Green → Yellow)  
- Geometry fallback mode  

---

# 📊 Ward-Level KPIs Generated

| Indicator | Scaling |
|------------|-----------|
| Bus Stops | per 1,000 population |
| Auto Stands | per 1,000 population |
| Taxi Stands | per 1,000 population |
| Metro Stations | per 1,000 population |
| Road Length | km per 1,000 population |
| Population (2025 Est.) | Area-share fallback |
| Intermodal Index | Composite 0–100 |

---

# 🏗 System Architecture

```
Field Data Collection
        ↓
Data Cleaning & Ward Normalization
        ↓
Population Scaling
        ↓
Infrastructure Density Metrics
        ↓
Min-Max Normalization
        ↓
Composite Index Engine
        ↓
Geospatial Rendering
        ↓
Downloadable Decision Dataset
```

---

# 🚀 Run Locally

```bash
git clone https://github.com/yourusername/kochi-transport-dashboard.git
cd kochi-transport-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

# 🎯 Why This Matters

Urban inequity hides in raw numbers.

Absolute counts distort policy.

Per-capita scaling reveals infrastructure imbalance.  
Weighted indices reveal modal dominance bias.  
Spatial overlays reveal concentrated under-service.

This system helps answer:

- Which wards are structurally neglected?
- What happens if metro investment weight increases?
- Is bus infrastructure compensating for metro gaps?
- Where should capital be allocated first?

---

# 📈 Policy Alignment

- SDG 11 – Sustainable Cities  
- SDG 9 – Infrastructure  
- Urban Resilience Diagnostics  
- Equity-Based Capital Allocation  
- Data-Driven Governance  

---

# 🔬 Grounded in Field Observations

Built after ward-level immersion in:

- Fort Kochi  
- Amravati  
- Nazarath  

Accessibility gaps and modal dependency patterns were converted into measurable infrastructure indicators.

---

# 🧩 Roadmap

<details>
<summary>Planned Extensions</summary>

- Accessibility Compliance Index (PWD standards)  
- Infrastructure Gini Coefficient  
- Climate Vulnerability Overlay  
- Multi-sector Sustainability Composite  
- Budget Allocation Simulation Engine  
- API-based Live Transport Feed Integration  

</details>

---

# 👤 Author

**Prakhar Kumar Rai**  
Urban Systems • Sustainability Analytics • Policy Modelling  

---

<div align="center">

### ⭐ If this project aligns with your work in urban governance or infrastructure analytics, feel free to connect.

</div>
