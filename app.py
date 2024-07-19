import streamlit as st
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        color: black;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: black;
    }
    .title {
    color: black;
    font-size: 32px;
    font-weight: 700;
    }
    .stTextArea label {
        font-size: 18px;
        font-weight: bold;
        color: black;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and sidebar
st.markdown('<h1 class="title">📝 NLP Application</h1>', unsafe_allow_html=True)
st.sidebar.title("🔍 Choose a Task")
task = st.sidebar.selectbox("Task", ["Sentiment Analysis", "Subjectivity Analysis", "Named Entity Recognition (NER)"])

st.write("""
    ### Instructions:
    1. Choose a task from the sidebar.
    2. Enter the text you want to analyze in the text area below.
    3. Click the **Analyze** button to see the results.
""")

text = st.text_area("Enter text to analyze", "Type here...")

@st.cache_resource
def load_model():
    model = spacy.load("en_core_web_sm")
    model.add_pipe("spacytextblob")
    return model

def analyze_sentiment(text, model):
    doc = model(text)
    polarity = doc._.polarity
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"

def analyze_subjectivity(text, model):
    doc = model(text)
    subjectivity = doc._.subjectivity
    if subjectivity > 0.5:
        return "Highly Subjective"
    elif subjectivity < 0.5:
        return "Less Subjective"
    else:
        return "Neutral"

def analyze_entities(text, model):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Load the model once
model = load_model()

# Analysis based on selected task
if st.button("Analyze"):
    if text.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        if task == "Sentiment Analysis":
            result = analyze_sentiment(text, model)
            st.subheader("Sentiment Analysis Result:")
            st.write(f"The sentiment of the text is **{result}**.")
        elif task == "Subjectivity Analysis":
            result = analyze_subjectivity(text, model)
            st.subheader("Subjectivity Analysis Result:")
            st.write(f"The text is **{result}**.")
        elif task == "Named Entity Recognition (NER)":
            entities = analyze_entities(text, model)
            st.subheader("Named Entity Recognition Result:")
            if entities:
                for ent in entities:
                    st.write(f"**Entity**: {ent[0]}, **Label**: {ent[1]}")
            else:
                st.write("No named entities found.")
