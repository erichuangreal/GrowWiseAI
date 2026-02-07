
## Feature Mapping: Auto vs Default vs Override


| Feature                                           | Source  | API/Default                                                                                                   |
| ------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| Elevation                                         | Auto    | Open-Elevation `api.open-elevation.com/api/v1/lookup`                                                         |
| Temperature                                       | Auto    | Open-Meteo `api.open-meteo.com/v1/forecast?current=temperature_2m,relative_humidity_2m`                       |
| Humidity                                          | Auto    | Open-Meteo (same call)                                                                                        |
| Soil_TN, Soil_TP, Soil_AP, Soil_AN                | Auto    | SoilGrids `rest.isric.org/soilgrids/v2.0/properties/query` (map nitrogen → TN; P from other layers or median) |
| Fire_Risk_Index                                   | Proxy   | Simple formula: `(1 - humidity/100) * (temp/40)` normalized to [0,1], or median                               |
| Slope                                             | Default | Median from training data (no simple free API for point slope)                                                |
| Menhinick_Index, Gleason_Index, Disturbance_Level | Default | Medians from [forest_health_data_with_target.csv](backend/forest_health_data_with_target.csv)                 |
