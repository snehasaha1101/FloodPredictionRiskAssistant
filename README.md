# 🌊 AI Flood Risk Prediction Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Google%20Gemini%20%2B%20LangChain-orange)
![SDG](https://img.shields.io/badge/SDG-13%20Climate%20Action-green)

An intelligent decision-support system that predicts localized flood risks in real-time and provides AI-generated safety protocols using Retrieval-Augmented Generation (RAG).

## 📌 Project Overview
Climate change has made flood events unpredictable. This project bridges the gap between raw weather data and public safety by:
1.  **Predicting Risk:** Using Machine Learning (Random Forest) to analyze rainfall, river levels, and soil moisture.
2.  **Guiding Action:** Using Generative AI (Google Gemini) to read emergency protocols and advise users on specific evacuation routes and safety measures.

**Problem Solved:** Bridges the "Decision Gap" where residents know it's raining but don't know *what to do* or *where to go*.

## 🚀 Features
* **Real-time Risk Analysis:** Instant classification of flood risk (Safe vs. High Danger) based on user inputs.
* **AI Safety Assistant:** A RAG-based chatbot that provides context-aware advice (e.g., specific shelter locations) based on the detected risk level.
* **Interactive Dashboard:** A clean, user-friendly web interface built with Streamlit.
* **Offline-Capable Embeddings:** Uses HuggingFace local embeddings to ensure stability and reduce API costs.

## 🛠️ Technologies Used
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **GenAI Framework:** LangChain, Google Gemini API (1.5 Flash)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## 📂 Project Structure
```bash
Flood-Risk-Assistant/
├── app.py                # Main application file
├── requirements.txt      # List of dependencies
├── protocols.txt         # Knowledge base for the AI Assistant
├── .streamlit/
│   └── secrets.toml      # (Ignored by Git) Stores API Keys securely
└── README.md             # Project Documentation
