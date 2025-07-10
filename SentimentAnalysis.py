import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load model 
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = load_model()

# App title 
st.title("Sentiment Analysis Web Application")
st.markdown("This application analyzes the sentiment of input sentences using a Hugging Face Transformer-Based Model.")

# User input text area
st.subheader("Enter sentences (one per line for better analysis)")
user_input = st.text_area("Input", height=250)

# Optional: user-defined limit for analysis
st.subheader("Number of sentences to analyze")
limit_input = st.text_input("Enter a number (leave blank to analyze all):", "")

# Button to start analysis
if st.button("Run Analysis"):
    if not user_input.strip():
        st.warning("Please enter at least one sentence.")
    else:
        sentences = [line.strip() for line in user_input.split("\n") if line.strip()]
        total_sentences = len(sentences)

        # Determine how many sentences to analyze
        try:
            limit = int(limit_input)
            if limit < 1 or limit > total_sentences:
                limit = total_sentences
        except:
            limit = total_sentences

        selected_sentences = sentences[:limit]

        # Run sentiment analysis
        results = analyzer(selected_sentences)

        # Prepare DataFrame
        df = pd.DataFrame({
            "Sentence": selected_sentences,
            "Sentiment": [res["label"] for res in results],
            "Confidence": [round(res["score"], 4) for res in results]
        })

        # Display DataFrame
        st.subheader(f"Sentiment Analysis Results ({len(selected_sentences)} of {total_sentences})")
        st.dataframe(df)

        # sentiment distribution plot
        st.subheader("Sentiment Distribution Curve")

        sentiment_counts = df["Sentiment"].value_counts().sort_index()
        labels = sentiment_counts.index.tolist()
        values = sentiment_counts.values

        x = np.arange(len(labels))
        x_smooth = np.linspace(x.min(), x.max(), 300)

        if len(x) > 1:
            z = np.polyfit(x, values, 3)
            p = np.poly1d(z)
            y_smooth = p(x_smooth)

            fig, ax = plt.subplots()
            ax.plot(x_smooth, y_smooth, color='mediumblue', linewidth=2)
            ax.scatter(x, values, color='darkorange', s=50)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title("Sentiment Distribution")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Number of Sentences")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Insufficient sentiment categories for curve visualization. At least two categories required.")

        # Summary section
        st.subheader("Summary")
        st.write(f"Total sentences entered: {total_sentences}")
        st.write(f"Sentences analyzed: {len(selected_sentences)}")
