import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import datetime

# Use a smaller and lighter model (distilbert instead of XLM-Roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to scale sentiment scores
def scale_score(label, score):
    return 5 * (score - 0.5) / 0.5 if label == "POSITIVE" else -5 * (1 - score) / 0.5

# Function to analyze sentiment in batches
def batch_analyze_sentiments(messages):
    results = sentiment_pipeline(messages)
    sentiments = [
        {"label": res["label"], "score": scale_score(res["label"], res["score"])}
        for res in results
    ]
    return sentiments

# Streamlit UI
st.title("Sentiment Analysis of Customer Conversations")

# Input section for customer service conversation
st.write("Enter a customer service conversation (each line is a new interaction between customer and service agent):")
conversation = st.text_area("Conversation", height=300, placeholder="Enter customer-service interaction here...")

# Add a button to run the analysis
if st.button('Run Sentiment Analysis'):
    if conversation:
        # Split conversation into separate messages (lines) for chunked processing
        messages = [msg.strip() for msg in conversation.split("\n") if msg.strip()]

        # Limit processing of large conversations (for memory optimization)
        MAX_MESSAGES = 20  # Only process up to 20 messages at once
        if len(messages) > MAX_MESSAGES:
            st.warning(f"Only analyzing the first {MAX_MESSAGES} messages for memory efficiency.")
            messages = messages[:MAX_MESSAGES]

        # Analyze each message for sentiment in batches
        sentiments = batch_analyze_sentiments(messages)

        # Create structured data
        results = []
        for i, msg in enumerate(messages):
            # Split each message into speaker and content
            if ": " in msg:
                speaker, content = msg.split(": ", 1)
            else:
                speaker, content = "Unknown", msg

            sentiment = sentiments[i]
            results.append({
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Speaker": speaker,
                "Message": content,
                "Sentiment": sentiment["label"],
                "Score": round(sentiment["score"], 2)
            })

        # Convert the results into a DataFrame
        df = pd.DataFrame(results)

        # Highlight positive and negative sentiments
        def style_table(row):
            if row["Sentiment"] == "POSITIVE":
                return ['background-color: #d4edda'] * len(row)
            elif row["Sentiment"] == "NEGATIVE":
                return ['background-color: #f8d7da'] * len(row)
            else:
                return [''] * len(row)

        styled_df = df.style.apply(style_table, axis=1)

        # Display the DataFrame
        st.write("Conversation with Sentiment Labels:")
        st.dataframe(styled_df)

        # Plot sentiment over time using Plotly
        fig = px.line(
            df,
            x='Timestamp',
            y='Score',
            color='Sentiment',
            title="Sentiment Score Over Time",
            markers=True
        )
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig)

    else:
        st.warning("Please enter a conversation before running the analysis.")