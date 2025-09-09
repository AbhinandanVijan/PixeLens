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