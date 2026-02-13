<div align="center">
🚏 Kochi Integrated Transport Intelligence Dashboard
Ward-Level Urban Analytics • Intermodal Index Engine • Policy Simulation System
<br>












<br>

From Fieldwork to Decision Intelligence
Built on real ward-level data from Kochi to model infrastructure equity, modal balance, and transport connectivity.

</div>
🌍 What This Is

A production-grade ward-level transport analytics system that:

Normalizes infrastructure per 1,000 population

Computes a customizable Intermodal Connectivity Index (0–100)

Ranks wards dynamically under different policy weights

Generates choropleth maps for spatial equity analysis

Exports decision-ready datasets

This is not a visualization tool.
It is a policy simulation engine for urban transport planning.

🧠 Core Engine: Intermodal Connectivity Index
Index=w1(Bus)+w2(Metro)+w3(Auto)+w4(Taxi)+w5(Inverse Distance)
Index=w
1
	​

(Bus)+w
2
	​

(Metro)+w
3
	​

(Auto)+w
4
	​

(Taxi)+w
5
	​

(Inverse Distance)

✔ Min-max normalized
✔ Weight-adjustable in real time
✔ Live rank recalculation
✔ Designed for capital allocation experiments

Adjust weights → Watch ward priorities shift → Identify structural bias.

🗺 Spatial Intelligence Layer

Ward shapefile integration

Interactive Folium choropleth

Auto-centering on selected wards

Custom sustainability color gradient

Fallback non-geometry preview mode

📊 Ward-Level KPIs Generated
Indicator	Scaling
Bus Stops	per 1,000 population
Auto Stands	per 1,000 population
Taxi Stands	per 1,000 population
Metro Stations	per 1,000 population
Road Length	km per 1,000 population
Population (2025 Est.)	Area-share fallback model
Intermodal Index	0–100 composite
🏗 Architecture Overview
Field Data (Transport + GIS)
        ↓
Ward Normalization
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

🚀 Run Locally
git clone https://github.com/yourusername/kochi-transport-dashboard.git
cd kochi-transport-dashboard
pip install -r requirements.txt
streamlit run app.py

🎯 Why This Project Matters

Urban transport inequity is invisible without normalization.

Raw counts mislead.
Per-capita scaling reveals concentration bias.
Weighted composite indices reveal infrastructure asymmetry.

This system helps answer:

Which wards are structurally under-served?

What happens if metro weight doubles?

Is bus infrastructure compensating for metro absence?

Where should capital be allocated first?

📈 Policy Alignment

✔ SDG 11 – Sustainable Cities
✔ SDG 9 – Infrastructure
✔ Equity-Based Capital Allocation
✔ Urban Resilience Diagnostics
✔ Governance Data Systems

🔬 Built From Field Observations

Grounded in on-site ward-level field immersion in:

Fort Kochi

Amravati

Nazarath

Infrastructure accessibility gaps and modal dependency patterns were observed and translated into measurable indicators (see Kochi Fieldwork Report).

🧩 Future Upgrades

Accessibility Compliance Index (PWD standards)

Climate Vulnerability Overlay

Infrastructure Gini Coefficient

Multi-sector Urban Sustainability Composite

API-based live transport feed integration

Budget allocation simulation module

👤 Author

Prakhar Kumar Rai
Urban Systems • Sustainability Analytics • Policy Modelling
Built as part of Bachelor’s in Analytics & Sustainability Studies
