# Captcha Recognition Project

This project implements an end-to-end captcha recognition solution that integrates both image and audio captcha processing. The image component uses EasyOCR to extract text from captcha images, while the audio component leverages a fine-tuned Wav2Vec2 model for converting audio captchas into text.

## Table of Contents

- [Environment Setup](#environment-setup)
- [How to Run the Solution](#how-to-run-the-solution)
- [High-Level Overview of the Approach](#high-level-overview-of-the-approach)
- [Assumptions and Design Decisions](#assumptions-and-design-decisions)
- [Dependencies](#dependencies)

## Environment Setup

1. **Google Colab / Local Setup**:
   - This project is designed to run in Google Colab, but you can also adapt it for a local environment.

2. **Mounting Google Drive**:
   - Since data files (images, audio, and processed DataFrames) are stored on Google Drive, ensure you mount your drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

3. **Install Dependencies**:
   - All required packages are listed in the `requirements.txt` file stored on your drive. Install them with:
     ```bash
     !pip install -r /content/drive/MyDrive/requirements.txt
     ```
   - This will install libraries such as librosa, torchaudio, noisereduce, EasyOCR, transformers, and others.

## How to Run the Solution

1. **Data Preparation**:
   - **Image Captcha Extraction**:
     - Use EasyOCR to extract text from images located in `/content/drive/MyDrive/captchaDatabase/captchas/images`.
     - The extracted text is stored in a CSV file: `/content/drive/MyDrive/captcha_dataset/extracted_image_captcha_data.csv`.
   - **Audio Data Preparation**:
     - Audio files are stored in `/content/drive/MyDrive/captchaDatabase/captchas/audio`.
     - Preprocess audio using functions that:
       - Resample audio to 16kHz.
       - Apply noise reduction, silence trimming, and volume normalization.
       - Augment data with random modifications (e.g., noise, time stretching, pitch shifting).
       - Preprocessed audio is saved as NumPy arrays.

2. **Data Splitting and Storage**:
   - The CSV with extracted image text is merged with corresponding audio file paths.
   - The combined dataset is split into training (80%) and testing (20%) sets.
   - Both datasets are further processed (e.g., converting audio waveforms to float32) and saved in compressed chunks (using pickle and gzip) for efficient loading.

3. **Model Preparation and Training**:
   - **Dataset Creation**:
     - Use the Hugging Face Datasets library to create training, evaluation, and testing datasets with two key fields: preprocessed audio (as sequences of float32) and `image_text`.
   - **Processor and Model**:
     - Load a pretrained Wav2Vec2 processor and model.
     - Map the dataset using a function that extracts audio features and tokenizes the text.
   - **Training Pipeline**:
     - A custom data collator handles dynamic padding of both audio inputs and text labels.
     - A custom Trainer class is implemented to compute the Connectionist Temporal Classification (CTC) loss.
     - Training is configured with early stopping, evaluation using Word Error Rate (WER), and periodic checkpointing.

4. **Evaluation and Inference**:
   - After training, the model’s performance is evaluated on the test set using WER.
   - The trained model and processor are saved for future inference.
   - A simple function is provided for transcribing new audio samples.

5. **Running Inference**:
   - Load the saved model and processor.
   - Pass audio samples through the transcription function to obtain text outputs.

## High-Level Overview of the Approach

- **Image Captcha Processing**:
  - Extract text from captcha images using EasyOCR.
  - Save the results in a CSV for further use.

- **Audio Captcha Processing**:
  - Standardize all audio to 16kHz and apply noise reduction and silence trimming.
  - Normalize volume and optionally apply random data augmentations.
  - Store preprocessed audio as NumPy arrays and merge with image captcha data.

- **Data Handling**:
  - Split the unified dataset into training and testing sets.
  - Save the DataFrames in compressed chunks to manage large datasets efficiently.

- **Model Training**:
  - Utilize Wav2Vec2 for audio-to-text conversion.
  - Create custom data collators and trainers to handle variable-length sequences and CTC loss.
  - Evaluate performance using Word Error Rate (WER) and make adjustments based on error analysis.

## Assumptions and Design Decisions

- **Data Consistency**:
  - The project assumes that the number of captcha images matches the number of corresponding audio files.
  - File naming conventions are assumed to be consistent for pairing images and audio.

- **Preprocessing Choices**:
  - Audio preprocessing includes aggressive noise reduction and data augmentation to mimic real-world conditions.
  - The decision to use only the first 0.5 seconds of audio for noise profiling is based on typical captcha characteristics.

- **Model and Training**:
  - Leveraging the pre-trained Wav2Vec2 model was chosen for its strong performance on speech recognition tasks.
  - A custom CTC loss function and data collator were implemented to better handle variable sequence lengths and padding issues.
  - Early stopping and checkpointing were used to prevent overfitting, given the limited amount of captcha data.

- **Storage and Efficiency**:
  - Large datasets are saved in smaller, compressed chunks to optimize memory usage during training.
  - Parallel processing was employed during audio preprocessing to speed up data preparation.

## Dependencies

- **Python (>=3.7)**
- **Google Colab (or a similar environment)**
- **Libraries**: librosa, torchaudio, noisereduce, pandas, numpy, soundfile, EasyOCR, transformers, datasets, evaluate, torch, joblib, etc.
