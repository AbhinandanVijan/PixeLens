import os
import platform

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if platform.system() == "Darwin":
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

import sqlite3
import numpy as np
import torch
import pickle
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import ViTFeatureExtractor, ViTModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Layer
import webbrowser
import socket


# ---------- Load Artifacts ----------
tokenizer = pickle.load(open("tokenizer.p", "rb"))
max_length = 32
# caption_model = load_model("vit_caption_model_final.h5", custom_objects={"PositionalEncoding": PositionalEncoding})
caption_model = load_model("vit_caption_model.h5")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# ---------- Initialize DB ----------
DB_FILE = "images.db"
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB NOT NULL,
                caption TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        conn.commit()

init_db()

# ---------- Helper Functions ----------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



def generate_caption_beam(model, tokenizer, photo, max_length, beam_width=3, start_token='start', end_token='end'):
    sequences = [[[tokenizer.word_index[start_token]], 0.0, []]]

    for _ in range(max_length):
        all_candidates = []

        for seq, score, words in sequences:
            if words and words[-1] == end_token:
                all_candidates.append((seq, score, words))
                continue

            padded_seq = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model.predict([photo, padded_seq], verbose=0)  # shape: (1, vocab_size)

            if preds.ndim != 2 or preds.shape[0] != 1:
                print(f"Unexpected prediction shape: {preds.shape}")
                continue

            vocab_logits = preds[0]  # shape: (vocab_size,)
            vocab_logits = np.array(vocab_logits, dtype=np.float64)
            vocab_logits[0] = -1.0  # optionally suppress <pad> token

            top_indices = np.argsort(vocab_logits)[-beam_width:]
            for idx in top_indices:
                word = word_for_id(idx, tokenizer)
                if word is None:
                    continue

                new_seq = seq + [idx]
                new_score = score - np.log(vocab_logits[idx] + 1e-9)
                new_words = words + [word]
                all_candidates.append((new_seq, new_score, new_words))

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

        if not sequences:
            print("Warning: No sequences left at step", _)
            return ""

    best_seq = sequences[0][2]
    caption = ' '.join([w for w in best_seq if w not in [start_token, end_token]])
    return caption.strip()



def saveImageDesc(img_path, caption, embedding):
    img = Image.open(img_path)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_data = img_bytes.getvalue()
    emb_blob = pickle.dumps(embedding)

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (image, caption, embedding) VALUES (?, ?, ?)", 
                       (img_data, caption, emb_blob))
        conn.commit()

def generate_caption(image_path):
    # try:
        if image_path is None:
            return "Please capture image by clicking on camera icon or upload an image before submitting."
        
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = vit_model(**inputs)
        photo = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().reshape(1, 768)

        caption = generate_caption_beam(caption_model, tokenizer, photo, max_length, 3)
        saveImageDesc(image_path, caption, photo.squeeze())
        return caption
    # except AttributeError as e:
    #     if "seek" in str(e) or "read" in str(e):
    #         return "Please click the camera icon to capture an image before submitting."
    #     else:
    #         return f"Unexpected attribute error: {e}"

    # except Exception as e:
    #     return f"An unexpected error occurred: {str(e)}"

def image_to_image_search(query_image, threshold=0.80):
    if isinstance(query_image, str):
        query_img = Image.open(query_image).convert("RGB")
    elif isinstance(query_image, Image.Image):
        query_img = query_image.convert("RGB")
    else:
        raise ValueError("Invalid input")

    inputs = feature_extractor(images=query_img, return_tensors="pt")
    with torch.no_grad():
        query_embedding = vit_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        query_embedding /= np.linalg.norm(query_embedding)

    results = []
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image, caption, embedding FROM images")
        for img_data, caption, emb_blob in cursor.fetchall():
            stored_embedding = pickle.loads(emb_blob)
            stored_embedding /= np.linalg.norm(stored_embedding)
            sim = cosine_similarity([query_embedding], [stored_embedding])[0][0]
            if sim >= threshold:
                results.append((sim, img_data, caption))

    results.sort(reverse=True, key=lambda x: x[0])
    top_results = results[:5]
    display_images = [(Image.open(BytesIO(img_bytes)), f"Sim: {sim:.2f} | {caption}") for sim, img_bytes, caption in top_results]
    return display_images

def search_images(keyword):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM images WHERE caption LIKE ?", ('%' + keyword + '%',))
        rows = cursor.fetchall()
    images = [Image.open(BytesIO(row[0])) for row in rows] if rows else []
    return gr.Gallery(images, columns=2)

# ---------- Gradio Interfaces ----------
caption_iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Image Caption Generator",
    description="Upload an image to generate a caption and store it locally."
)

search_text_iface = gr.Interface(
    fn=search_images,
    inputs="text",
    outputs=gr.Gallery(),
    title="Search Images by Text",
    description="Search images based on their generated captions."
)

search_image_iface = gr.Interface(
    fn=image_to_image_search,
    inputs=[
        gr.Image(type="filepath", label="Query Image"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.80, label="Similarity Threshold")
    ],
    outputs=gr.Gallery(),
    live=True,
    title="Search Images by Image",
    description="Upload an image and filter similar results by cosine similarity."
)

# ---------- Launch ----------
if __name__ == "__main__":
    interface = gr.TabbedInterface(
        [caption_iface, search_text_iface, search_image_iface],
        ["Generate Caption", "Search by Text", "Search by Image"]
    )
    interface.launch(share=False, inbrowser=True)

