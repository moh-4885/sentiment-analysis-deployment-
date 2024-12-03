import streamlit as st
import requests
import json

def predict_sentiment(texts):
    # API endpoint (adjust if your FastAPI is running on a different host/port)
    url = " https://sentiment-anlyses-go745vrj9-moh-4885s-projects.vercel.app/predict"
    
    # Prepare the payload
    payload = {"texts": texts}
    
    try:
        # Send POST request to the API
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Sentiment Analysis", page_icon=":mag:")
    
    # Title of the application
    st.title("📊 Sentiment Analysis Tool")
    
    # Text input for multiple lines
    text_input = st.text_area(
        "Enter text(s) to analyze (one per line)",
        height=200,
        placeholder="Enter your text here..."
    )
    
    # Predict button
    if st.button("Analyze Sentiment"):
        # Check if input is not empty
        if text_input.strip():
            # Split input into lines
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            
            # Predict sentiment
            results = predict_sentiment(texts)
            
            # Display results
            if results:
                st.subheader("Analysis Results:")
                for result in results:
                    # Determine sentiment label
                    sentiment_label = "Positive 😊" if result['sentiment'] == 1 else "Negative 😞"
                    
                    # Create columns for text and sentiment
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(result['text'])
                    with col2:
                        st.write(f"Sentiment: {sentiment_label}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()