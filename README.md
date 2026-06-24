---
title: Sneakerhead RAG
emoji: 👟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
python_version: '3.11'
app_file: app.py
pinned: false
---

# 👟 Sneakerhead RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about sneakers, brands, and sneaker culture.

## Features

- Ask questions about sneaker brands (Nike, Adidas, Puma)
- Learn about iconic models (Air Jordan, Yeezy, Air Max )
- Get information about sneaker history and culture
- Friendly, knowledgeable AI assistant

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your OpenAI API key to `.env`
4. Run the app: `python app.py`

## Data Source

The knowledge base includes information from the `documents/` folder containing text files about sneaker brands and models.

## Hugging Face Deployment

Add these secrets in Hugging Face:
- `OPENAI_API_KEY`

## License

MIT