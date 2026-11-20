# Field Drainage Using LiDAR
### *Intelligent LiDAR-Based Agricultural Field Drainage Planning System*

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![WhiteboxTools](https://img.shields.io/badge/Hydro_Engine-WhiteboxTools-green.svg)](https://www.whiteboxgeo.com/)
[![GeoPandas](https://img.shields.io/badge/Vectors-GeoPandas-teal.svg)](https://geopandas.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Engine-ReportLab-orange.svg)](https://www.reportlab.com/)

> **Notice:** This project is proprietary and protected by copyright. See [LICENSE](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/LICENSE) for terms. All rights reserved.

---

## Executive Summary
Surface waterlogging is one of the primary drivers of yield loss, delayed sowing, root asphyxiation, and secondary soil salinization in Indian agriculture—particularly across clay-rich Vertisols (Black Cotton Soils) of Maharashtra, Madhya Pradesh, Gujarat, and the canal-irrigated command areas of Punjab, Haryana, and Andhra Pradesh. 

**Field Drainage Using LiDAR** is an intelligent, automated, 100% free-to-build geospatial decision-support system designed to plan precise surface drainage networks for agricultural holdings. By processing high-resolution LiDAR point clouds (`.las`/`.laz`) and Digital Elevation Models (DEMs), the system automatically:
1. Filters bare-earth terrain and builds hydrologically conditioned Digital Terrain Models (DTMs).
2. Delineates micro-catchments, D8 flow directions, flow accumulation pathways, and Topographic Wetness Indices (TWI).
3. Detects genuine waterlogging depressions while filtering high-frequency DEM sensor noise.
4. Extracts existing farm ditches and bridges severed culvert connections using the peer-reviewed **Roelens et al. (2018)** morphological workflow.
5. Recommends optimal gravity-fed surface trench alignments to route standing water out of fields into natural drainage outlets.
6. Delivers an interactive web GIS dashboard (Streamlit + Folium) and an automated, investor/agronomist-ready PDF Drainage Plan Report.

---

## Key Capabilities & Features

- **Multi-Format Terrain Ingestion:** Ingests unclassified or classified LiDAR point clouds (`.las`, `.laz`) via `laspy` and raster elevation models (GeoTIFF, Copernicus 30m, CartoDEM) via `rasterio`.
- **Hydrological Conditioning & Sink Discrimination:** Distinguishes genuine ponding depressions from DEM artifacts using depth, area, and persistence filtering before selective breach depressions or sink filling.
- **Morphological Ditch Extraction (Roelens et al., 2018):** Implements Relative Elevation Attribute (REA), Otsu automatic thresholding, directional morphological thinning (skeletonization), lowest-LiDAR-point snapping, and obstacle/culvert connectivity correction.
- **Gravity-Fed Route Optimization:** Identifies natural pour points / field discharge boundaries and computes minimum-cost, positive-slope ditch alignments to minimize earthwork volume.
- **Interactive Dual-Pane GIS Dashboard:** Interactive Streamlit interface featuring layer toggling (DTM, Hillshade, Slope, Flow Accumulation, Ditches, Waterlogging Hotspots, Recommended Routes) and side-by-side swipe comparison.
- **Automated Field Drainage Plan Report:** Generates publication-grade PDF engineering reports detailing field slope distribution, ponding risk metrics, proposed cut-and-fill ditch lengths, and geo-referenced cartographic maps.

---

## Strict 3-Tier Free Data Strategy & Disclaimer

Access to aerial LiDAR is strictly restricted or non-existent for routine smallholder plots in India. To guarantee complete integrity, reproducibility, and 100% free execution without fabricated claims, this system operates on a transparent **3-Tier Data Architecture**:

```
+-----------------------------------------------------------------------------------+
|                           3-TIER DATA STRATEGY                                    |
+-----------------------------------------------------------------------------------+
| Tier 1: Satellite Macro DEM (Free Public Data)                                    |
| - Source: Copernicus DEM GLO-30 (30m) & ISRO Bhuvan CartoDEM                      |
| - Scope: Watershed delineation, regional macro-slope, valley context.             |
| - Disclaimer: Explicitly labeled as SATELLITE DEM (cannot resolve micro-ditches). |
+-----------------------------------------------------------------------------------+
| Tier 2: Real Sub-Meter LiDAR Benchmark (Free Public Data)                          |
| - Source: OpenTopography / USGS 3DEP (US/European Agricultural Testbeds)          |
| - Scope: Algorithmic validation of sub-meter point cloud bare-earth extraction,   |
|   Roelens ditch extraction, and centimeter-scale waterlogging sinks.              |
+-----------------------------------------------------------------------------------+
| Tier 3: Synthetic Indian Farm Model (Programmatically Generated)                  |
| - Source: In-house programmatic NumPy/SciPy synthetic terrain generator.          |
| - Scope: Simulates typical 2-to-5 hectare Indian agricultural fields featuring    |
|   elevated field bunds (daul/medh), micro-depressions, and clogged boundary      |
|   trenches with known ground truth for deterministic validation.                   |
+-----------------------------------------------------------------------------------+
```

---

## 100% Free Technology Stack

| Layer | Free Open-Source Component | Replaces (Commercial / Paid) |
|---|---|---|
| **Language & Environment** | Python 3.10+, NumPy, SciPy | MATLAB, SAS |
| **Point Cloud Processing** | `laspy` (with `lazrs` / `laszip`), PDAL | TerraScan, FugroViewer, Global Mapper |
| **Raster & Vector GIS** | `rasterio`, `geopandas`, `shapely`, `pyproj` | ArcGIS Pro, ArcPy, ERDAS IMAGINE |
| **Hydrological Engine** | `whitebox` (WhiteboxTools Open Core) | ArcGIS Spatial Analyst / ArcHydro |
| **Computer Vision / Morph** | `scikit-image`, `scikit-learn` | Commercial Photogrammetry Toolkits |
| **Web GIS Dashboard** | `streamlit`, `folium`, `streamlit-folium` | ArcGIS Online, Web AppBuilder |
| **Report Generation** | `reportlab`, `matplotlib` | ArcGIS Map Series, Paid Reporting APIs |
| **External Verification** | QGIS (v3.28+ LTR) | Commercial Desktop GIS |

---

## Documentation Structure

Comprehensive technical specifications and implementation blueprints are cataloged in the [`docs/`](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs) directory:

1. [**01: Ideation & Requirements**](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs/01_IDEATION_AND_REQUIREMENTS.md)  
   *Problem breakdown in Indian agriculture, user personas, functional/non-functional requirements, strict v1 boundaries, and QGIS verification protocols.*
2. [**02: System Architecture**](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs/02_SYSTEM_ARCHITECTURE.md)  
   *Six-module decoupled architecture, complete directory structure, headless GIS vs. UI separation, and rigorous Coordinate Reference System (CRS) metric management.*
3. [**03: Algorithms & Data Flow**](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs/03_ALGORITHMS_AND_DATAFLOW.md)  
   *End-to-end processing pipeline, sequence diagrams, Whitebox hydrological modeling, Roelens et al. (2018) ditch extraction, and depression vs. noise filtering.*
4. [**04: Tech Stack & Data Strategy**](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs/04_TECH_STACK_AND_DATA_STRATEGY.md)  
   *Detailed tooling justifications, API comparisons, and step-by-step acquisition guides for Copernicus GLO-30, OpenTopography, and the Synthetic Indian Farm generator.*
5. [**05: Development Roadmap**](file:///C:/Users/Varad%20D/.gemini/antigravity/scratch/field-drainage-lidar/docs/05_DEVELOPMENT_ROADMAP.md)  
   *Six-phase development plan with acceptance criteria, unit test milestones, and complete Streamlit UI layout and wireframe specifications.*

---

## Quick Start (Preview & Exploration)

```bash
# Clone the repository (Private access only)
git clone https://github.com/varadrz/field-drainage-lidar.git
cd field-drainage-lidar

# Setup Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt

# Run the GIS Dashboard
streamlit run app.py
```

---

## Intellectual Property & License
Copyright (c) 2026 Varad Patil. All Rights Reserved.  
This software, documentation, and architecture are proprietary. Strictly no permission is granted for copying, modifying, merging, publishing, distributing, sublicensing, or selling any part of this system without express prior written consent from the author.
