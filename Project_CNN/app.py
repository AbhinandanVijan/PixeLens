import sqlite3
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Database file
DB_FILE = "images.db"

# Create table if not exists
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB NOT NULL,
                caption TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

# Save image to SQLite
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    caption = request.form['caption']

    if file:
        filename = secure_filename(file.filename)
        image_data = file.read()  # Read image as binary

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO images (image, caption) VALUES (?, ?)", (image_data, caption))
            conn.commit()

        return jsonify({"message": "Image stored successfully"})

# Search images by keyword
@app.route('/search', methods=['GET'])
def search_images():
    keyword = request.args.get('keyword')

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM images WHERE caption LIKE ?", ('%' + keyword + '%',))
        images = [row[0] for row in cursor.fetchall()]

    return jsonify({"images_count": len(images), "images": images})

if __name__ == '__main__':
    app.run(debug=True)
