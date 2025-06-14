import streamlit as st
import nltk
from transformers import pipeline
import torch
import textstat
from datetime import datetime
import pyperclip
import io
from docx import Document
import base64

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the pipelines
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    sentiment = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    return summarizer, sentiment

# Load the pipelines
summarizer, sentiment_analyzer = load_pipelines()

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This app helps you summarize long texts using AI. 
    
    ### Features:
    - AI-powered text summarization
    - Adjustable summary length
    - Summary statistics
    - Support for long texts
    - Text analysis
    - Export options
    """)
    
    st.markdown("---")
    
    # Summary length slider in sidebar
    max_length = st.slider(
        "Maximum summary length (words):",
        min_value=50,
        max_value=500,
        value=150,
        step=10
    )
    
    # Summary style selection
    summary_style = st.radio(
        "Summary Style:",
        ["Standard", "Bullet Points", "Detailed"]
    )

# Main content
st.title("📝 Text Summarizer")

# Create two columns for text input and summary
col1, col2 = st.columns(2)

# Text input in first column
with col1:
    st.subheader("Input Text")
    text_input = st.text_area("Enter your text here:", height=400)
    
    # Text Analysis Section
    if text_input:
        st.subheader("📊 Text Analysis")
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            # Basic stats
            word_count = len(text_input.split())
            reading_time = textstat.reading_time(text_input)
            st.metric("Word Count", word_count)
            st.metric("Reading Time", f"{reading_time} minutes")
            
        with col_analysis2:
            # Sentiment analysis
            try:
                sentiment = sentiment_analyzer(text_input[:512])[0]  # Analyze first 512 chars
                st.metric("Sentiment", sentiment['label'])
                st.metric("Confidence", f"{sentiment['score']:.2%}")
            except:
                st.info("Sentiment analysis not available for this text")

# Summary in second column
with col2:
    st.subheader("Summary")
    if st.button("Generate Summary", use_container_width=True):
        if text_input:
            with st.spinner("Generating summary..."):
                try:
                    # Split text into chunks if it's too long
                    max_chunk_length = 1024
                    chunks = [text_input[i:i + max_chunk_length] for i in range(0, len(text_input), max_chunk_length)]
                    
                    summaries = []
                    for chunk in chunks:
                        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                    
                    final_summary = " ".join(summaries)
                    
                    # Format summary based on selected style
                    if summary_style == "Bullet Points":
                        sentences = nltk.sent_tokenize(final_summary)
                        final_summary = "\n".join([f"• {sentence}" for sentence in sentences])
                    elif summary_style == "Detailed":
                        final_summary = f"Detailed Summary:\n\n{final_summary}"
                    
                    # Display the summary
                    st.write(final_summary)
                    
                    # Display summary statistics
                    original_words = len(text_input.split())
                    summary_words = len(final_summary.split())
                    compression_ratio = (1 - (summary_words / original_words)) * 100
                    
                    st.info(f"""
                    📊 Summary Statistics:
                    - Original text: {original_words} words
                    - Summary: {summary_words} words
                    - Compression ratio: {compression_ratio:.1f}%
                    """)
                    
                    # Export options
                    st.subheader("Export Options")
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        if st.button("📋 Copy to Clipboard"):
                            pyperclip.copy(final_summary)
                            st.success("Copied to clipboard!")
                    
                    with col_export2:
                        # Create a download button for the summary
                        doc = Document()
                        doc.add_paragraph(final_summary)
                        docx_bytes = io.BytesIO()
                        doc.save(docx_bytes)
                        docx_bytes.seek(0)
                        b64 = base64.b64encode(docx_bytes.read()).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="summary.docx">📥 Download as DOCX</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Transformers") 