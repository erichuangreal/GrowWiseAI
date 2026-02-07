# cxc_hackathon

Tree Species Survivability Prediction Project

## Dataset Overview

- **Total datapoints:** 2,783
- **Tree species:** 4 (Acer saccharum, Prunus serotina, Quercus alba, Quercus rubra)
- **Overall mortality rate:** 57.05%

## Environmental Conditions

### Light Conditions:
- Light_ISF (light intensity: 0.032-0.161)
- Light_Cat (category: Low/Med/High)

### Soil Conditions:
- Soil (7 soil types)
- Sterile (Sterile/Non-Sterile)

### Mycorrhizal Fungi:
- AMF (Arbuscular Mycorrhizal Fungi: 0-100%)
- EMF (Ectomycorrhizal Fungi: 0-100%)
- Myco (mycorrhizal type: AMF/EMF)
- SoilMyco (soil mycorrhizal community)

### Species Interaction:
- Conspecific (Conspecific/Heterospecific/Sterilized)

### Time:
- Time (days in study: 14-115.5)

## Target Variable

- **Event:** 0 = survived, 1 = died

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the parser:
```bash
python3 parse_tree_data.py
```