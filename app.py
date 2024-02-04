import streamlit as st
import pickle
import TextCleanUp as tcu



tfidf = pickle.load(open('./model/vectorizer_tfidf.pkl','rb'))
model = pickle.load(open('./model/model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = tcu.get_complete_text_clean_up(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")