# PixeLens

PixeLens is an AI-powered image captioning and semantic search platform. It allows users to upload images, generate natural language captions using Vision Transformers, and perform real-time text-to-image and image-to-image searches.

## Features
- **Image Captioning**: Generates descriptive captions for user-uploaded images via Vision Transformer models.
- **Semantic Search**: Supports both text-to-image and image-to-image retrieval using cosine similarity on image embeddings.
- **Lightweight Storage**: Stores images, captions, and embeddings in a fast, searchable SQLite database.
- **Real-Time Results**: Delivers search results instantly for a smooth user experience.

## Tech Stack
- **Backend**: Python, Gradio, Vision Transformers (ViT)
- **Database**: SQLite
- **Search**: Cosine similarity on precomputed embeddings
- **Frontend**: Gradio

## Installation
```bash
git clone https://github.com/yourusername/pixelens.git
cd pixelens
pip install -r requirements.txt
