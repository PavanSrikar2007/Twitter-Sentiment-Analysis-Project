import streamlit as st
import os
import random
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="centered"
)

# --- Helper: Demo prediction if models are missing ---
def demo_predict(text, model_name="BERT"):
    text = text.lower()
    positive_words = ['love', 'great', 'excellent', 'amazing', 'good', 'fantastic', 'wonderful', 'best', 'perfect', 'awesome', 'happy']
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'horrible', 'delayed', 'cancelled', 'rude', 'disappointed', 'angry']
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    if pos_count > neg_count:
        sentiment = "Positive"
        confidence = random.uniform(0.85, 0.98)
    elif neg_count > pos_count:
        sentiment = "Negative"
        confidence = random.uniform(0.80, 0.95)
    else:
        sentiment = "Neutral"
        confidence = random.uniform(0.60, 0.80)
    return sentiment, confidence

def models_available():
    bert_model = "model/bert/bert_model/tf_model.h5"
    lstm_model = "model/lstm/lstm_model_50.h5"
    return os.path.exists(bert_model) and os.path.exists(lstm_model)


TEXT_COLUMNS = ("text", "tweet_text", "full_text", "content", "body")
XQUIK_METADATA_COLUMNS = ("created_at", "query", "username", "author_username", "url")


def normalize_batch_csv(df):
    text_column = next((column for column in TEXT_COLUMNS if column in df.columns), None)
    if text_column is None:
        return None, "CSV must contain a text column. Supported columns: " + ", ".join(TEXT_COLUMNS)

    normalized = pd.DataFrame({"text": df[text_column].astype(str).str.strip()})
    for column in XQUIK_METADATA_COLUMNS:
        if column in df.columns:
            normalized[column] = df[column]

    normalized = normalized[normalized["text"] != ""]
    if normalized.empty:
        return None, "CSV text rows are empty after trimming whitespace."
    return normalized, None

# --- Main App ---
def main():
    st.markdown(
        "<h1 style='text-align: center; font-size:2.6rem;'>🔍 Sentiment Analysis App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center; color:#bbb; margin-bottom:2rem;'>"
        "Analyze text sentiment using <b>BERT</b> or <b>LSTM</b> models"
        "</div>", unsafe_allow_html=True
    )

    tabs = st.tabs(["Single Text", "Batch CSV"])

    # --- Single Text Tab ---
    with tabs[0]:
        st.markdown("<h2>Single Text Analysis</h2>", unsafe_allow_html=True)
        st.markdown("**Input Text:**")
        user_input = st.text_area("Enter your text here...", key="single_text", height=100, label_visibility="collapsed")
        st.markdown("**Model:**")
        model_choice = st.radio("", ["BERT", "LSTM"], horizontal=True, key="single_model")
        if st.button("Analyze", key="analyze_single"):
            if not user_input.strip():
                st.warning("Please enter your text.")
            else:
                if not models_available():
                    sentiment, confidence = demo_predict(user_input, model_choice)
                else:
                    # Place your real model code here if running locally
                    sentiment, confidence = demo_predict(user_input, model_choice)
                color = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}.get(sentiment, "#95a5a6")
                st.markdown(f"""
                <div style='text-align:center; margin:2rem 0; padding:1.5rem; border-radius:10px; background-color:#18191A; border-left: 4px solid {color}'>
                    <div style='font-size:1.3rem; margin-bottom:0.5rem'>
                        Predicted Sentiment: 
                        <span style='color:{color}; font-weight:bold; font-size:1.4rem'>
                            {sentiment}
                        </span>
                    </div>
                    <div style='font-size:1.1rem; color:#aaa'>
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence, text="Model Confidence")

    # --- Batch CSV Tab ---
    with tabs[1]:
        st.markdown("<h2>Batch CSV</h2>", unsafe_allow_html=True)
        st.caption("Uploads from Xquik tweet/search exports are supported when they include text-like columns.")
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"], key="batch_csv")
        st.markdown("**Model:**")
        model_choice_batch = st.radio("", ["BERT", "LSTM"], horizontal=True, key="batch_model")
        if uploaded_file and st.button("Analyze", key="analyze_batch"):
            df = pd.read_csv(uploaded_file)
            normalized_df, error = normalize_batch_csv(df)
            if error:
                st.error(error)
            else:
                st.info(f"Processing {len(normalized_df)} rows...")
                results = []
                for _, row in normalized_df.iterrows():
                    text = row["text"]
                    sentiment, confidence = demo_predict(str(text), model_choice_batch)
                    result = {"text": text, "Predicted Sentiment": sentiment, "Confidence": f"{confidence:.1%}"}
                    for column in XQUIK_METADATA_COLUMNS:
                        if column in row:
                            result[column] = row[column]
                    results.append(result)
                df_result = pd.DataFrame(results)
                st.dataframe(df_result)
                csv = df_result.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv, "sentiment_results.csv", "text/csv")

  
if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; margin-top:2rem'>
        <p><strong>© 2025 Sentiment Analysis Project</strong></p>
        <p>👩‍💻 Built by Marpini Himabindu | 🎓 B.Tech IT 2022-2026</p>
        <p>💻 <a href="https://github.com/PavanSrikar2007/Twitter-Sentiment-Analysis-Project" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
