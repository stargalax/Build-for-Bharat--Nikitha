# 🌾 AI Agricultural Intelligence Dashboard

An interactive Streamlit app that allows users to explore **Indian crop production data** with AI-powered insights and visualizations.

---

## 🚀 Overview

This system combines official Indian agricultural datasets with AI (Google Gemini) to provide:

- Trend analysis of crop production over years
- State-wise and crop-wise comparisons
- Top crops by production for a given state or district
- Actionable insights for farmers and policymakers
- Interactive visualizations using Plotly

The system supports multiple input types, including text queries like:

- “Top crops in Karnataka”
- “Compare Rice production between Punjab and Andhra Pradesh in 2004”
- “Show Wheat production trend in Punjab”

---

## 📦 Features

- **AI-powered query understanding:** Uses Google Gemini to parse natural language queries.
- **Dynamic data fetching:** Pulls data from `data.gov.in` API and falls back to public CSV via CKAN if API fails.
- **Visualizations:** Bar charts, line charts, and tables powered by Plotly.
- **Trend analysis:** Aggregates production data by year, state, or crop for trend visualization.
- **Metrics summary:** Total production, average, maximum, and record counts displayed.
- **Conversation context:** Supports simple chat history to provide contextual responses.

---

## 🛠️ Tech Stack

- **Frontend & Interaction:** [Streamlit](https://streamlit.io/)
- **Data Visualization:** [Plotly](https://plotly.com/python/)
- **Data Source:** [data.gov.in Crop Production Dataset](https://data.gov.in/resources/district-wise-crop-production-statistics)
- **AI Integration:** [Google Gemini](https://developers.google.com/experimental/generative-ai)
- **Python Libraries:** `pandas`, `requests`, `io`, `time`, `plotly`, `json`

---

## 📄 Data Sources

- District-wise Crop Production Statistics, Ministry of Agriculture & Farmers Welfare
- Public CSV via CKAN
