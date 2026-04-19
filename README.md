# SentimentLens 🔍

A sentiment analysis web app powered by a fine-tuned DistilBERT model, built with Streamlit.

## Demo🔗 **[Live App](https://sentiment-analyzer-erkjwmgj2wfdydgwkwatpb.streamlit.app/)**

[App Screenshot](https://github.com/abdurhaq/Sentiment-Analyzer/blob/main/Screenshot%202026-04-19%20224459.png)

## What it does
Paste any text — a review, tweet, or headline — and the model instantly predicts whether the sentiment is **positive** or **negative**, along with a confidence score and probability breakdown.

## Tech Stack
- **Model** — DistilBERT fine-tuned on Stanford SST-2 dataset (67,000 sentences)
- **Framework** — PyTorch + HuggingFace Transformers
- **Frontend** — Streamlit
- **Accuracy** — ~91% on SST-2 validation set

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/sentiment-lens.git
cd sentiment-lens
```

**2. Install dependencies**
```bash
pip install transformers datasets evaluate torch streamlit accelerate
```

**3. Train the model**
```bash
python train.py
```

**4. Launch the app**
```bash
streamlit run app.py
```

## Project Structure
```
sentiment-lens/
├── train.py          # Fine-tuning script
├── app.py            # Streamlit web app
├── sentiment_model/  # Saved model (generated after training)
└── README.md
```

## How it works
1. DistilBERT (a lighter, faster version of BERT) is loaded with pre-trained weights
2. A classification head is added and fine-tuned on SST-2 for 3 epochs
3. The trained model is served via a Streamlit interface that returns predictions in real time
