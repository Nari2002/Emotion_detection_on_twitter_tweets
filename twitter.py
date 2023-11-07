import joblib
import sklearn
import numpy as np
import pandas as pd
import streamlit as st


model = joblib.load("C:/Users/soppoju narender/Desktop/FS-DataScience/NLP/text_classification_on_twitter_tweets/twitter_emotional_model.pkl")
tfidi = joblib.load("C:/Users/soppoju narender/Desktop/FS-DataScience/NLP/text_classification_on_twitter_tweets/tfidf_vectorizer_twitter_data.pkl")

def analysis(input_text):
    input_data_features = tfidi.transform(input_text)
    data_features = pd.DataFrame(input_data_features.toarray())
 
    prediction = model.predict(data_features)
    print(prediction)
    if (prediction[0] == 0):
        return "Sadness"
    elif (prediction[0] == 1):
        return "Neutral"
    elif (prediction[0] == 2):
        return "worry"
    elif (prediction[0] == 3):
        return "surprise"
    elif (prediction[0] == 4):
        return "love"
    elif (prediction[0] == 5 or 7):
        return "fun or Happiness"
    elif (prediction[0] == 6):
        return "hate"
   
    else:
        return "relief"

def main():
    st.markdown("""
<style>
    /* Change the font size for all text within the Streamlit app */
    body {
        font-size: 40px;
    }
</style>
""", unsafe_allow_html=True)
    def set_bg_hack_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0Azdqw1ivMrJG51xVioDkUeSBxBql8HVM1UU671J5S88MjsB7ARm8RbGC8lCG6-TjDN0&usqp=CAU");
                 background-size: cover
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    set_bg_hack_url()
    st.title("Emotion Detection on twitter tweets")
    input_text = st.text_input("Enter a tweet :thinking_face: :thinking_face:")
    
    dig =""
    if st.button("Detect my emotion 	:hugging_face:"):
        dig = analysis([input_text])
    st.success(dig)
        


if __name__ == '__main__':
    main()