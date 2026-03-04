# GrowWiseAI

> AI-powered forest health prediction system using machine learning and real-time environmental data

## Links
- Live Demo: https://growwiseai.erichuangreal.dev/
- Devpost: https://devpost.com/software/growwiseai

# GrowWise AI

**AI-powered decision support for forest health and ecological resilience**

GrowWise AI is an intelligent environmental analytics platform that uses machine learning to predict forest health outcomes based on ecological and environmental conditions.  

Our goal is to transform raw environmental data into actionable insights that support reforestation, conservation, and climate resilience efforts.

---

## Features

- Predict forest health using environmental variables  
- Assess the impact of fire risk on ecosystem stability  
- Analyze soil and climate factors affecting forest conditions  
- AI-powered decision support for sustainable land management  

---

## Model Inputs & Output

### Independent Variables (Environmental Factors)

- Elevation  
- Temperature  
- Humidity  
- Soil Total Nitrogen (`Soil_TN`)  
- Soil Total Phosphorus (`Soil_TP`)  
- Fire Risk Index  

### Dependent Variable

- **Health Status** (Forest condition classification)

---

## Dataset

We trained our model using the:

**Forest Health and Ecological Diversity Dataset**  
https://www.kaggle.com/datasets/ziya07/forest-health-and-ecological-diversity

---

## Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Google Generative AI API
- Scikit-learn

### Frontend
- React
- Vite
- Node.js

---

## Local Development Setup

### 1️Backend

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install requests python-dotenv google-generativeai
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8001
```

### 2️Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```
