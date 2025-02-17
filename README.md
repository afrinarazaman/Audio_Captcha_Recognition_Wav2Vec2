# Audio Captcha Recognition (Wav2Vec2)

This project implements an end-to-end captcha recognition solution that integrates both image and audio captcha processing. The image component uses **EasyOCR** to extract text from captcha images, while the audio component leverages a fine-tuned **Wav2Vec2** model for converting audio captchas into text.

## Table of Contents
- [Environment Setup](#environment-setup)
- [How to Run the Solution](#how-to-run-the-solution)
- [High-Level Overview of the Approach](#high-level-overview-of-the-approach)
- [Assumptions and Design Decisions](#assumptions-and-design-decisions)
- [Dependencies](#dependencies)

## Environment Setup

### Google Colab / Local Setup
- This project is designed to run in Google Colab, but it can also be adapted for a local environment.

### Mounting Google Drive
Since data files (images, audio, and processed DataFrames) are stored on Google Drive, ensure you mount your drive:

```python
from google.colab import drive
drive.mount('/content/drive')
