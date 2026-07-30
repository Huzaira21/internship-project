"""Unit tests for data_loader.py and inference.py"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
from src.data_loader import load_captions, clean_caption, preprocess_captions, get_image_path
from src.inference import compute_bleu


# ---- Tests for data_loader.py ----

def test_load_captions_returns_dataframe():
    df = load_captions("data/captions.txt")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_captions_has_expected_columns():
    df = load_captions("data/captions.txt")
    assert "image" in df.columns
    assert "caption" in df.columns


def test_clean_caption_lowercases_text():
    result = clean_caption("A Dog Is RUNNING", stop_words=set())
    assert all(word.islower() for word in result)


def test_clean_caption_removes_punctuation():
    result = clean_caption("A dog, running!", stop_words=set())
    joined = " ".join(result)
    assert "," not in joined
    assert "!" not in joined


def test_clean_caption_handles_empty_string():
    result = clean_caption("", stop_words=set())
    assert result == []


def test_clean_caption_removes_stop_words():
    result = clean_caption("a dog is running", stop_words={"a", "is"})
    assert "a" not in result
    assert "is" not in result


def test_preprocess_captions_adds_expected_columns():
    df = pd.DataFrame({"caption": ["A dog is running.", "A cat sleeps."]})
    processed = preprocess_captions(df)
    assert "tokens" in processed.columns
    assert "cleaned_caption" in processed.columns


def test_get_image_path_builds_correct_path():
    path = get_image_path("data/Images", "test.jpg")
    assert path == os.path.join("data/Images", "test.jpg")


# ---- Tests for inference.py ----

def test_compute_bleu_identical_sentences_scores_high():
    score = compute_bleu("a dog is running", ["a dog is running"])
    assert score > 0.9


def test_compute_bleu_completely_different_scores_low():
    score = compute_bleu("a dog is running", ["completely unrelated text here"])
    assert score < 0.3


def test_compute_bleu_handles_empty_generated_caption():
    score = compute_bleu("", ["a dog is running"])
    assert score >= 0.0


def test_compute_bleu_handles_multiple_references():
    score = compute_bleu("a dog runs", ["a dog is running", "a dog runs fast", "a cat sleeps"])
    assert score >= 0.0