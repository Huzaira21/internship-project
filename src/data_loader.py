"""Data loading and preprocessing utilities for the Flickr8k dataset."""

import pandas as pd
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_captions(captions_path):
    """Load the raw captions CSV into a DataFrame."""
    return pd.read_csv(captions_path)


def clean_caption(text, stop_words):
    """Lowercase, strip punctuation, tokenize, and remove stop-words."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def preprocess_captions(df, caption_col="caption"):
    """Apply cleaning to an entire captions DataFrame."""
    stop_words = set(stopwords.words('english'))
    df = df.copy()
    df['tokens'] = df[caption_col].apply(lambda x: clean_caption(x, stop_words))
    df['cleaned_caption'] = df['tokens'].apply(lambda x: ' '.join(x))
    return df


def get_image_path(images_dir, image_name):
    """Return the full path to an image file."""
    return os.path.join(images_dir, image_name)