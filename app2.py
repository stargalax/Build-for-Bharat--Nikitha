import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import json
import time
import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"Loaded API Key: '{DATA_GOV_API_KEY}'")
DATASET_METADATA = {
"dataset_title": "District-wise Crop Production Statistics",
"dataset_url": "https://data.gov.in/catalog/district-wise-crop-production-statistics",
"resource_id": "35be999b-0208-4354-b557-f6ca9a5355de",
"source_api": "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de",
"provider": "Ministry of Agriculture & Farmers Welfare, Government of India",
"last_updated": "2024"
}
# Page config
st.set_page_config(
    page_title="AI Agricultural Chat",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: white;
        color: #333;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize APIs


# Configure Gemini - USE CORRECT MODEL NAME
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Using latest stable flash model

# API Helper Functions
# def fetch_crop_data(filters: Dict, limit: int = 100, debug: bool = False) -> pd.DataFrame:
#     """Fetch crop data from data.gov.in with pagination - FIXED WITH CORRECT FIELD NAMES"""
#     #url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
#     url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    
#     all_records = []
#     max_iterations = 50
#     iteration = 0
    
#     while iteration < max_iterations:
#         params = {
#             "api-key": CROP_API_KEY,
#             "format": "json",
#             "limit": limit,
#             "offset": iteration * limit
#         }
        
#         # Use correct lowercase field names with underscores
#         for key, value in filters.items():
#             params[f"filters[{key}]"] = value
        
#         if debug:
#             st.write(f"**API Request URL:** {url}")
#             st.write(f"**Filters:** {filters}")
#             st.write(f"**Full params:** {params}")
            
#         try:
#             response = requests.get(url, params=params, timeout=30)
#             response.raise_for_status()
#             data = response.json()
            
#             if debug:
#                 st.write(f"**Response status:** {response.status_code}")
#                 st.write(f"**Records found:** {len(data.get('records', []))}")
#                 if data.get('records'):
#                     st.write(f"**Sample record:** {data['records'][0]}")
            
#             if 'records' in data and len(data['records']) > 0:
#                 all_records.extend(data['records'])
#                 if len(data['records']) < limit:
#                     break
#             else:
#                 break
                
#             iteration += 1
#             time.sleep(0.1)
            
#         except Exception as e:
#             st.error(f"API Error: {e}")
#             if debug:
#                 st.write(f"**Error details:** {str(e)}")
#             break
    
#     if debug:
#         st.write(f"**Total records fetched:** {len(all_records)}")
    
#     if all_records:
#         df = pd.DataFrame(all_records)
#         # Clean production_ column (note the underscore!)
#         if 'production_' in df.columns:
#             df['production_'] = pd.to_numeric(df['production_'], errors='coerce')
#         if 'area_' in df.columns:
#             df['area_'] = pd.to_numeric(df['area_'], errors='coerce')
#         return df
#     return pd.DataFrame()
@st.cache_data
def fetch_crop_data(filters: Dict, limit: int = 100, debug: bool = False) -> pd.DataFrame:
    """
    Fetch crop production data from the District-wise Crop Production Statistics dataset.
    Primary: official API (with key)
    Fallback: CKAN CSV export (public access)
    """
    import requests, pandas as pd, time, io
    import streamlit as st

    api_url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
    fallback_url = "https://data.gov.in/node/3283761/datastore/export/csv"
   
    api_key = os.getenv("DATA_GOV_API_KEY")

    all_records = []
    iteration = 0
    max_iterations = 20
    
    try:
        # --- Try official API ---
        while iteration < max_iterations:
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": limit,
                "offset": iteration * limit
            }

            for key, value in filters.items():
                params[f"filters[{key}]"] = value

            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code in [401, 403]:
                raise PermissionError("Official API restricted → switching to CSV fallback.")

            response.raise_for_status()
            data = response.json()

            if 'records' in data and len(data['records']) > 0:
                all_records.extend(data['records'])
                if len(data['records']) < limit:
                    break
            else:
                break

            iteration += 1
            time.sleep(0.1)

        if not all_records:
            raise ValueError("No data returned from API.")

        df = pd.DataFrame(all_records)

    except (PermissionError, ValueError, requests.RequestException):
        # --- Fallback to CSV dump ---
        st.warning("⚠️ API access failed — using public CSV fallback.")
        try:
            csv_data = requests.get(fallback_url, timeout=60)
            csv_data.raise_for_status()
            df = pd.read_csv(io.StringIO(csv_data.text))
        except Exception as e:
            st.error(f"Fallback CSV failed: {e}")
            return pd.DataFrame()

    # --- Clean numeric + year columns ---
    for col in ['area_', 'production_']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'crop_year' in df.columns:
        df['crop_year'] = pd.to_numeric(df['crop_year'], errors='coerce')

    if debug:
        st.write(f"✅ Records: {len(df)}")
        st.dataframe(df.head())

    return df

# def fetch_crop_data(filters: Dict, limit: int = 100, debug: bool = False) -> pd.DataFrame:
#     """
#     Fetch crop production data from the original dataset:
#     District-wise Crop Production Statistics.
    
#     Primary source: API (requires whitelisted key)
#     Fallback: CKAN JSON dump (open access)
#     """
#     import requests, pandas as pd, time
#     import streamlit as st

#     api_url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de"
#     fallback_url = "https://data.gov.in/node/3283761/datastore/export/json"
#     api_key = "579b464db66ec23bdd00000199e0dad63c24493a7afc337902e81dc0"

#     all_records = []
#     max_iterations = 50
#     iteration = 0

#     try:
#         # --- Attempt official API first ---
#         while iteration < max_iterations:
#             params = {
#                 "api-key": api_key,
#                 "format": "json",
#                 "limit": limit,
#                 "offset": iteration * limit
#             }

#             for key, value in filters.items():
#                 params[f"filters[{key}]"] = value

#             if debug:
#                 st.write("🔗 Official API Request:", api_url)
#                 st.write(params)

#             response = requests.get(api_url, params=params, timeout=30)

#             # Handle authorization errors
#             if response.status_code in [401, 403]:
#                 raise PermissionError("Official API access restricted — using fallback source.")

#             response.raise_for_status()
#             data = response.json()

#             if 'records' in data and len(data['records']) > 0:
#                 all_records.extend(data['records'])
#                 if len(data['records']) < limit:
#                     break
#             else:
#                 break

#             iteration += 1
#             time.sleep(0.1)

#         if not all_records:
#             raise ValueError("No records returned from API.")

#     except (PermissionError, ValueError, requests.RequestException) as e:
#         st.warning(f"⚠️ {e}")
#         st.info("Switching to fallback (CKAN public JSON dump)...")

#         try:
#             response = requests.get(fallback_url, timeout=60)
#             response.raise_for_status()
#             all_records = response.json()
#         except Exception as e2:
#             st.error(f"Fallback failed: {e2}")
#             return pd.DataFrame()

#     # --- Convert to DataFrame ---
#     df = pd.DataFrame(all_records)

#     # --- Clean numeric columns ---
#     for col in ['area_', 'production_']:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#     # --- Debug view ---
#     if debug:
#         st.write(f"✅ Total Records: {len(df)}")
#         st.dataframe(df.head())

#     return df


def analyze_query_with_llm(user_query: str, conversation_history: List) -> Dict:
    """Use Gemini to understand user intent and extract parameters"""
    
    prompt = f"""You are an AI assistant helping with Indian agricultural data queries. 

Analyze this user query and extract the following in JSON format:
- query_type: "comparison", "trend", "ranking", "single_state", or "general"
- states: list of state names (use proper capitalization: "Tamil Nadu", "Uttar Pradesh", etc.)
- crops: list of crop names (use proper capitalization: "Rice", "Wheat", "Potato", etc.)
- years: list of years or year range as numbers
- analysis_type: "production", "area", "yield", or "general"
- needs_chart: true/false
- chart_type: "bar", "line", "table", or null

Common Indian states: Karnataka, Tamil Nadu, Maharashtra, Punjab, Uttar Pradesh, Madhya Pradesh, Bihar, West Bengal, Gujarat, Rajasthan, Andhra Pradesh, Telangana, Kerala, Odisha, Haryana

Common crops: Rice, Wheat, Cotton, Sugarcane, Maize, Bajra, Jowar, Groundnut, Soyabean, Gram, Tur, Urad, Moong, Potato

User Query: "{user_query}"

Conversation Context: {conversation_history[-3:] if conversation_history else "None"}

Return ONLY valid JSON, no other text."""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        return result
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return {
            "query_type": "general",
            "states": [],
            "crops": [],
            "years": [],
            "analysis_type": "general",
            "needs_chart": False,
            "chart_type": None
        }

def generate_insight(query_params: Dict, data: pd.DataFrame) -> str:
    """Use Gemini to generate natural language insights"""
    
    # Prepare data summary with CORRECT field name
    data_summary = ""
    if not data.empty:
        if 'production_' in data.columns:
            total_prod = data['production_'].sum()
            avg_prod = data['production_'].mean()
            data_summary = f"Total Production: {total_prod:,.0f} tonnes, Average: {avg_prod:,.0f} tonnes, Records: {len(data)}"
    
    prompt = f"""Based on the agricultural data analysis, provide a clear, concise insight in 2-3 sentences.

Query Type: {query_params.get('query_type')}
States: {query_params.get('states')}
Crops: {query_params.get('crops')}
Years: {query_params.get('years')}

Data Summary: {data_summary}

Provide actionable insights for farmers or policymakers. Be specific and data-driven."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Analysis complete. Please review the data visualization above."

def create_visualization(query_params: Dict, data: pd.DataFrame):
    """Create appropriate visualization based on query type - FIXED WITH CORRECT FIELD NAMES"""
    
    if data.empty:
        st.warning("No data available for visualization")
        return
    
    # Use correct field names (all lowercase with underscores)
    prod_col = 'production_'
    state_col = 'state_name'
    crop_col = 'crop'
    year_col = 'crop_year'
    district_col = 'district_name'
    
    chart_type = query_params.get('chart_type', 'bar')
    query_type = query_params.get('query_type', 'unknown')
    
    if chart_type == 'line' and year_col in data.columns and prod_col in data.columns:
        # Trend analysis - line chart with TOTAL production per year
        df_clean = data[[year_col, prod_col]].copy()
        df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        if state_col in data.columns:
            df_clean[state_col] = data[state_col]
            # Sum by year and state for trends
            df_grouped = df_clean.groupby([year_col, state_col])[prod_col].sum().reset_index()
            fig = px.line(df_grouped, x=year_col, y=prod_col, color=state_col,
                         title="Total Production Trend Over Years",
                         labels={prod_col: "Total Production (tonnes)", year_col: "Year"},
                         markers=True)
        elif crop_col in data.columns:
            df_clean[crop_col] = data[crop_col]
            # Sum by year and crop for trends
            df_grouped = df_clean.groupby([year_col, crop_col])[prod_col].sum().reset_index()
            fig = px.line(df_grouped, x=year_col, y=prod_col, color=crop_col,
                         title="Total Production Trend Over Years",
                         labels={prod_col: "Total Production (tonnes)", year_col: "Year"},
                         markers=True)
        else:
            # Sum by year for overall trends
            df_grouped = df_clean.groupby(year_col)[prod_col].sum().reset_index()
            fig = px.line(df_grouped, x=year_col, y=prod_col,
                         title="Total Production Trend Over Years",
                         labels={prod_col: "Total Production (tonnes)", year_col: "Year"},
                         markers=True)
        
        fig.update_layout(height=500, template="plotly_white", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
    elif query_type == 'comparison' and state_col in data.columns and prod_col in data.columns:
        # For comparisons - show TOTAL production per state
        df_clean = data[[state_col, prod_col]].copy()
        df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        # Sum production by state
        df_grouped = df_clean.groupby(state_col)[prod_col].sum().reset_index()
        df_grouped = df_grouped.sort_values(prod_col, ascending=False)
        
        fig = px.bar(df_grouped, x=state_col, y=prod_col,
                    title="Total Production Comparison Between States",
                    labels={prod_col: "Total Production (tonnes)", state_col: "State"},
                    color=state_col,
                    text=prod_col)
        
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(height=500, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    elif (query_type == 'single_state' or query_type == 'top') and crop_col in data.columns and prod_col in data.columns:
        # For top crops - show TOTAL production per crop
        df_clean = data[[crop_col, prod_col]].copy()
        df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
        df_clean = df_clean.dropna()
        
        # Sum production by crop and get top 10
        df_grouped = df_clean.groupby(crop_col)[prod_col].sum().reset_index()
        df_grouped = df_grouped.sort_values(prod_col, ascending=False).head(10)
        
        fig = px.bar(df_grouped, x=crop_col, y=prod_col,
                    title="Top Crops by Total Production",
                    labels={prod_col: "Total Production (tonnes)", crop_col: "Crop"},
                    color=prod_col,
                    color_continuous_scale='Greens',
                    text=prod_col)
        
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(height=500, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == 'bar' or chart_type is None:
        # Default bar chart logic
        if state_col in data.columns and crop_col in data.columns and prod_col in data.columns:
            df_clean = data[[state_col, crop_col, prod_col]].copy()
            df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
            df_clean = df_clean.dropna()
            df_grouped = df_clean.groupby([state_col, crop_col])[prod_col].sum().reset_index()
            
            fig = px.bar(df_grouped, x=state_col, y=prod_col, color=crop_col,
                        title="Production Comparison",
                        labels={prod_col: "Total Production (tonnes)", state_col: "State"},
                        text=prod_col)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        elif crop_col in data.columns and prod_col in data.columns:
            df_clean = data[[crop_col, prod_col]].copy()
            df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
            df_clean = df_clean.dropna()
            df_grouped = df_clean.groupby(crop_col)[prod_col].sum().reset_index()
            df_grouped = df_grouped.sort_values(prod_col, ascending=False).head(10)
            
            fig = px.bar(df_grouped, x=crop_col, y=prod_col,
                        title="Top Crops by Total Production",
                        labels={prod_col: "Total Production (tonnes)", crop_col: "Crop"},
                        color=prod_col,
                        color_continuous_scale='Greens',
                        text=prod_col)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        elif district_col in data.columns and prod_col in data.columns:
            df_clean = data[[district_col, prod_col]].copy()
            df_clean[prod_col] = pd.to_numeric(df_clean[prod_col], errors='coerce')
            df_clean = df_clean.dropna()
            df_grouped = df_clean.groupby(district_col)[prod_col].sum().reset_index()
            df_grouped = df_grouped.sort_values(prod_col, ascending=False).head(10)
            
            fig = px.bar(df_grouped, x=district_col, y=prod_col,
                        title="Top Districts by Total Production",
                        labels={prod_col: "Total Production (tonnes)", district_col: "District"},
                        color=prod_col,
                        color_continuous_scale='Blues',
                        text=prod_col)
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        else:
            st.warning("Cannot create bar chart with available data")
            return
        
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Show summary statistics
    if prod_col in data.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        total_prod = data[prod_col].sum()
        avg_prod = data[prod_col].mean()
        max_prod = data[prod_col].max()
        num_records = len(data)
        
        with col1:
            st.metric("Total Production", f"{total_prod:,.0f} tonnes")
        with col2:
            st.metric("Average Production", f"{avg_prod:,.0f} tonnes")
        with col3:
            st.metric("Maximum Production", f"{max_prod:,.0f} tonnes")
        with col4:
            st.metric("Total Records", f"{num_records:,}")
    
    # Show data table (limited to 20 rows)
    with st.expander("📋 View Raw Data"):
        st.dataframe(data.head(20), use_container_width=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Header
st.title("🌾 AI Agricultural Intelligence")
st.markdown("<p class='subtitle'>Ask me anything about Indian crop production data</p>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"<div class='chat-message {role_class}'>{message['content']}</div>", 
                unsafe_allow_html=True)
    
    # Display chart if exists
    if message["role"] == "assistant" and "chart_data" in message:
        create_visualization(message["query_params"], message["chart_data"])

# Chat input
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("Ask a question...", 
                               placeholder="e.g., Compare rice production in Punjab vs Tamil Nadu in 2020",
                               key="user_input",
                               label_visibility="collapsed")
with col2:
    send_button = st.button("Send 📤")

# Example questions
st.markdown("### 💡 Try asking:")
example_col1, example_col2, example_col3 = st.columns(3)
with example_col1:
    if st.button("🌾 Top crops in Karnataka"):
        user_input = "Show me top 5 crops in Karnataka for 2000"
        send_button = True
with example_col2:
    if st.button("📊 Compare states"):
        user_input = "Compare Rice production between Karnataka and Andhra Pradesh in 2004"
        send_button = True
with example_col3:
    if st.button("📈 Yearly trends"):
        user_input = "Show Wheat production trend in Punjab"
        send_button = True

# Process user input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🤔 Analyzing your query..."):
        # Analyze query with LLM
        query_params = analyze_query_with_llm(user_input, st.session_state.conversation_context)
        
        # Fetch data based on parameters - USE CORRECT FIELD NAMES
        data = pd.DataFrame()
        
        if query_params.get('query_type') != 'general':
            filters = {}
            
            states = query_params.get('states', [])
            crops = query_params.get('crops', [])
            years = query_params.get('years', [])
            
            # For trend analysis or multi-year queries
            if query_params.get('query_type') == 'trend' and years:
                all_data = []
                for year in years:
                    year_filters = {"crop_year": str(year)}
                    if states:
                        year_filters["state_name"] = states[0]
                    if crops:
                        year_filters["crop"] = crops[0]
                    
                    year_data = fetch_crop_data(year_filters, debug=st.session_state.debug_mode)
                    if not year_data.empty:
                        all_data.append(year_data)
                
                if all_data:
                    data = pd.concat(all_data, ignore_index=True)
            
            # For comparisons or single queries
            else:
                if states:
                    if len(states) > 1:
                        all_data = []
                        for state in states:
                            state_filters = {"state_name": state}
                            if crops:
                                state_filters["crop"] = crops[0]
                            if years:
                                state_filters["crop_year"] = str(years[0])
                            
                            state_data = fetch_crop_data(state_filters, debug=st.session_state.debug_mode)
                            if not state_data.empty:
                                all_data.append(state_data)
                        
                        if all_data:
                            data = pd.concat(all_data, ignore_index=True)
                    else:
                        filters["state_name"] = states[0]
                
                if crops and len(states) <= 1:
                    filters["crop"] = crops[0]
                
                if years and len(states) <= 1:
                    filters["crop_year"] = str(years[0])
                
                if filters and len(states) <= 1:
                    data = fetch_crop_data(filters, debug=st.session_state.debug_mode)
        
        # Generate response
        if not data.empty:
            insight = generate_insight(query_params, data)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": insight,
                "query_params": query_params,
                "chart_data": data
            })
            
            st.session_state.conversation_context.append({
                "query": user_input,
                "params": query_params
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I couldn't find data for your query. Here's what I searched for:\n\nStates: {states}\nCrops: {crops}\nYears: {years}\n\nTry different parameters or check if the data exists for those years."
            })
    
    st.rerun()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI-powered system uses:
    - **Google Gemini** for natural language understanding
    - **data.gov.in** for agricultural data
    - **Plotly** for interactive charts
    
    ### What you can ask:
    - Compare production between states
    - Find top crops in a state
    - Analyze yearly trends
    - Rank districts by production
    
    ### Available Data:
    - Years: 1997-2023
    - States: All Indian states
    - Crops: Rice, Wheat, Cotton, Sugarcane, Maize, Potato, etc.
   
    ### Note to judging panel:
    - I gave more emphasis on the visualization part of the project as per the role's title
    - This project helped me understand the real problem with current dataset and forced my to look for solutions that can handle the differnced in the API data allowing me to explore different paths
     """)
     
    st.markdown("---")
    
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Data source: Ministry of Agriculture & Farmers Welfare via data.gov.in")
