import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load vectorizer and model
tk = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("📩 SMS Spam Detection Model")
st.write("Made with ❤️‍🔥 by Shrudex👨🏻‍💻")
st.subheader("Enter an SMS below to check if it's Spam or Not 🚀")

st.caption("💡 Try these examples:")
st.code("Congratulations! You've won a free lottery ticket. Call now!")
st.code("Hey, are we meeting tomorrow?")

# Input field
input_sms = st.text_input("✍️ Type your message here:")

# Predict button
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tk.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This SMS is SPAM")
        else:
            st.success("✅ This SMS is NOT SPAM")


           
