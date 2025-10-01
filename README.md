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
cd pixelens/Code

# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate it
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Step 3: Install requirements
pip install -r requirements.txt

# Step 4: Launch using run.sh
./run.sh



##############
1. Additional .ipynb files and scripts are kept in additional_code_files folder
2. Image_Caption_Gen_lstm_final_pipeline.ipynb is the final model code
3. augment_images.py is the script to augment images
4.images.db is a SQLITE db, it will be created automatically if not present
5. vit_caption_model.h5 is the final model weights file
6. tokenizer.p is required for the pipeline to work
pip install -r requirements.txt
