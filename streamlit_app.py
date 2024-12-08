import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def preprocess(text):
    """
    Placeholder for text preprocessing.
    In a real-world scenario, you might want to add more advanced preprocessing 
    like lowercasing, removing punctuation, etc.
    """
    return text

def load_models():
    """
    Load the vectoriser and logistic regression model from pickle files.
    Replace the file paths with the actual paths to your saved model files.
    """
    # Load the vectoriser
    with open('vectoriser-ngram-(1,2).pickle', 'rb') as file:
        vectoriser = pickle.load(file)
    
    # Load the LR Model
    with open('Sentiment-LR.pickle', 'rb') as file:
        LRmodel = pickle.load(file)
    
    return vectoriser, LRmodel

def predict_sentiment(vectoriser, model, texts):
    """
    Predict sentiment for given texts using the loaded model and vectoriser
    
    Args:
        vectoriser: Fitted CountVectorizer
        model: Trained Logistic Regression model
        texts: List of texts to analyze
    
    Returns:
        List of tuples with (text, sentiment)
    """
    # Preprocess texts
    preprocessed_texts = [preprocess(text) for text in texts]
    
    # Vectorize the texts
    textdata = vectoriser.transform(preprocessed_texts)
    
    # Predict sentiments
    sentiment = model.predict(textdata)
    
    # Combine texts with their sentiments
    results = []
    for text, pred in zip(texts, sentiment):
        results.append({
            'text': text,
            'sentiment': pred
        })
    
    return results

def main():
    # Load models at app startup
    try:
        vectoriser, LRmodel = load_models()
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'vectoriser-ngram-(1,2).pickle' and 'Sentiment-LR.pickle' are in the correct directory.")
        return

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
            try:
                results = predict_sentiment(vectoriser, LRmodel, texts)
                
                # Display results
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
            
            except Exception as e:
                st.error(f"An error occurred during sentiment analysis: {e}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()