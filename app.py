import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Cache NLTK download and model loading
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    vectorizer = joblib.load('vectorizer.pkl')
    classifier = joblib.load('classifier.pkl')
    return vectorizer, classifier

# Cache preprocessing function
@st.cache_data
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    review = ' '.join([
        ps.stem(word) for word in review.split() 
        if word not in set(all_stopwords)
    ])
    
    return review

# Cache review predictions
@st.cache_data
def predict_review(review):
    processed_review = preprocess_text(review)
    X_test = vectorizer.transform([processed_review]).toarray()
    return classifier.predict(X_test)[0]

# Load resources
vectorizer, classifier = load_resources()

# Maintain review history
if 'review_history' not in st.session_state:
    st.session_state.review_history = []


# [Previous caching and preprocessing functions remain the same]

def show_visualizations():
    st.title("Overall Review Visualization")
    if len(st.session_state.review_history) > 0:
        # Create DataFrame from history
        df = pd.DataFrame(st.session_state.review_history)
        
        # Sentiment Counts
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)
        
        # Donut Chart (in first column)
        with col1:
            fig1 = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index, 
                values=sentiment_counts.values, 
                hole=0.3,  # Creates the ring effect
                marker_colors=['blue', 'yellow']
            )])
            fig1.update_layout(
                title='Sentiment Distribution 📈',
                title_x=0,  # Center the title
                height=400  # Adjust height for better display
            )
            st.plotly_chart(fig1)
        
        # Bar Chart (in second column)
        with col2:
            fig2 = px.bar(
                x=sentiment_counts.index, 
                y=sentiment_counts.values, 
                title='Review Sentiment Overview 📊',
                labels={'x':'Sentiment', 'y':'Number of Reviews'},
                color=sentiment_counts.index,
                color_discrete_map={'Positive':'green', 'Negative':'red'}
            )
            fig2.update_layout(
                title_x=0,  # Center the title
                height=400  # Adjust height for better display
            )
            st.plotly_chart(fig2)
    else:
        st.text("Kindly enter review for visualization")  

# Detailed test cases in tabular form
def show_test_cases():
    st.title("Inputs you can try")
    
    detailed_cases = [
        {
            'Review': "Not a good experience at all",
            'Description': "Negative review with explicit negation",
            'Expected Sentiment': "Negative"
        },
        {
            'Review': "Excellent food, amazing service",
            'Description': "Strongly positive review with multiple positive words",
            'Expected Sentiment': "Positive"
        },
        {
            'Review': "Terrible restaurant with poor quality",
            'Description': "Strongly negative review with multiple negative descriptors",
            'Expected Sentiment': "Negative"
        },
        {
            'Review': "Average meal, nothing special",
            'Description': "Neutral leaning towards negative review",
            'Expected Sentiment': "Negative"
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(detailed_cases)
    
    # Add Actual Sentiment column
    df['Actual Sentiment'] = df['Review'].apply(lambda x: 'Positive' if predict_review(x) == 1 else 'Negative')
    
    # Highlight matching/mismatching sentiments
    def color_sentiment(val):
        return 'background-color: green' if val == 'Correct' else 'background-color: red'
    
    # Compute sentiment match
    df['Match'] = (df['Expected Sentiment'] == df['Actual Sentiment']).map({True: 'Correct', False: 'Incorrect'})
    
    # Display table with styling
    styled_df = df.style.applymap(color_sentiment, subset=['Match'])
    st.dataframe(styled_df)

# Add footer with credits
def add_footer():
    st.markdown("---")
    st.markdown("👨‍💻 Developed by Sandeep S", unsafe_allow_html=True)

# Main function
def main():
    st.title("🍽️👨‍🍳Restaurant Review Sentiment Analyzer🥤🍔")

    # Review input
    st.title("User's Review")
    user_review = st.text_area("Enter your restaurant review:", "")

    # Predict button
    if st.button("Predict Sentiment"):
        if user_review.strip():
            # Predict review
            prediction = predict_review(user_review)
            
            # Store in history
            st.session_state.review_history.append({
                'review': user_review, 
                'sentiment': 'Positive' if prediction == 1 else 'Negative'
            })
            
            # Display result
            if prediction == 1:
                st.success("😊 Positive Review - The customer liked it!")
            else:
                st.error("☹️ Negative Review - The customer didn't like it!")
            
            # Visualizations
            show_visualizations()
        else:
            st.warning("Please enter a review first!")

# Run the app
if __name__ == "__main__":
    main()
    show_test_cases()
    add_footer()