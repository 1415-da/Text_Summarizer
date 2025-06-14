# Text Summarizer Web App

A web application built with Streamlit that uses AI to summarize long texts. The app features an adjustable summary length slider and provides summary statistics.

## Features

- Text summarization using BART model
- Adjustable summary length
- Summary statistics (word count, compression ratio)
- Support for long texts through chunking
- Modern and user-friendly interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

To run the application, use the following command:
```bash
streamlit run app.py
```

The app will open in your default web browser. If it doesn't, you can access it at `http://localhost:8501`.

## Usage

1. Paste your text in the input area
2. Adjust the summary length using the slider
3. Click "Generate Summary"
4. View your summary and statistics

## Requirements

- Python 3.7+
- Streamlit
- NLTK
- Transformers
- PyTorch

## Note

The first time you run the app, it will download the BART model which might take a few minutes depending on your internet connection. 