import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Flood Risk AI", page_icon="🌊")

# Securely load API Key from secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found. Please set it in .streamlit/secrets.toml")

# ==========================================
# PART 1: MACHINE LEARNING ENGINE (Prediction)
# ==========================================
@st.cache_resource
def build_prediction_model():
    """Trains a Random Forest model on simulated historical data."""
    # Simulated Dataset (Rainfall mm, River Level m, Soil Moisture %)
    data = {
        'rainfall': [10, 100, 15, 200, 120, 5, 300, 50, 250, 90],
        'river_level': [1.2, 3.5, 1.3, 4.5, 3.8, 1.1, 5.5, 2.0, 4.8, 2.9],
        'soil_moisture': [30, 80, 35, 90, 85, 25, 95, 45, 92, 70],
        'flood_label': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # 0=Safe, 1=Flood
    }
    df = pd.DataFrame(data)
    
    X = df[['rainfall', 'river_level', 'soil_moisture']]
    y = df['flood_label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# PART 2: RAG ENGINE (The Assistant)
# ==========================================
@st.cache_resource
def build_rag_system():
    """Reads protocols.txt and sets up the AI Assistant."""
    try:
        # 1. Load the text file
        if not os.path.exists("protocols.txt"):
            st.warning("⚠️ 'protocols.txt' file not found. AI Assistant will be disabled.")
            return None 
            
        loader = TextLoader("protocols.txt")
        documents = loader.load()
        
        # 2. Embeddings (Local - Free & Unlimited)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = Chroma.from_documents(documents, embeddings)
        
        # 3. LLM Setup (Gemini 1.5 Flash)
        # Note: We use "gemini-1.5-flash" as it is the standard stable version
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        return {"vector_store": vector_store, "llm": llm}
        
    except Exception as e:
        st.error(f"RAG Setup Error: {e}")
        return None

def get_ai_advice(rag_system, risk_level, user_query="What should I do?"):
    """Queries the RAG system for specific advice based on risk level."""
    if not rag_system:
        return "System Error: protocols.txt not found or API Key missing."

    # Retrieve relevant info
    docs = rag_system["vector_store"].similarity_search(risk_level, k=2)
    context_text = "\n".join([d.page_content for d in docs])
    
    # Prompt Engineering
    prompt = (
        f"You are a Flood Safety Officer. The current Risk Level is {risk_level}.\n"
        f"Based ONLY on the following protocols, answer the user's question.\n"
        f"Protocols: {context_text}\n\n"
        f"User Question: {user_query}\n"
        f"Answer (keep it concise and actionable):"
    )
    
    response = rag_system["llm"].invoke(prompt)
    return response.content

# ==========================================
# PART 3: USER INTERFACE (Streamlit)
# ==========================================
def main():
    st.title("🌊 AI Flood Risk Assistant")
    st.markdown("Enter environmental conditions to predict flood risk and receive AI-guided safety protocols.")

    # A. Sidebar Inputs
    st.sidebar.header("Sensor Data")
    rain = st.sidebar.slider("Rainfall (mm)", 0, 400, 50)
    river = st.sidebar.slider("River Level (m)", 0.0, 10.0, 2.0)
    soil = st.sidebar.slider("Soil Moisture (%)", 0, 100, 40)

    # Load Systems
    model = build_prediction_model()
    rag_system = build_rag_system()

    # B. Main Action
    if st.button("Analyze Current Conditions", type="primary"):
        
        # 1. Predict Risk
        prediction = model.predict([[rain, river, soil]])[0]
        probs = model.predict_proba([[rain, river, soil]])[0]
        
        # 2. Display Result
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Risk Status")
            if prediction == 1:
                st.error("🚨 HIGH RISK")
                risk_status = "HIGH RISK"
                confidence = probs[1]
            else:
                st.success("✅ LOW RISK")
                risk_status = "LOW RISK"
                confidence = probs[0]
            st.metric("Confidence", f"{confidence*100:.1f}%")

        # 3. AI Assistant Advice (RAG)
        with col2:
            st.subheader("🤖 Assistant Protocol")
            if rag_system:
                with st.spinner("Consulting emergency protocols..."):
                    # We ask the AI specifically about immediate actions
                    advice = get_ai_advice(rag_system, risk_status, "What are the immediate actions and shelter locations?")
                    st.info(advice)
            else:
                st.warning("AI Assistant unavailable (Check API Key or protocols.txt).")

if __name__ == "__main__":
    main()