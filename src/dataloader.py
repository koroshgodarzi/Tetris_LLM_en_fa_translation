import requests
from itertools import islice
import zipfile
import os
from torch.utils.data import Dataset
import torch
import string
import re
from torch.utils.data import random_split, DataLoader


def download_opus_subtitles(number_of_samples_for_train=10000):
    """
    Downloads the en-fa OpenSubtitles zip file, extracts it, 
    and renames the English and Farsi files
    to have a .txt extension.
    """
    zip_url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-fa.txt.zip"
    zip_filename = "en-fa.txt.zip"
    extracted_dir = "." # Extract to the current directory

    # 1. Download the zip file using requests
    print(f"Downloading {zip_url}...")
    try:
        response = requests.get(zip_url, stream=True)
        response.raise_for_status() 
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return


    # 2. Extract the zip file using zipfile
    print(f"Extracting {zip_filename} using zipfile...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print("Extraction complete.")
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        return

    # 3. Rename the extracted files
    old_en_filename = os.path.join(extracted_dir, "OpenSubtitles.en-fa.en")
    new_en_filename = os.path.join(extracted_dir, "OpenSubtitles.en-fa.en.txt")
    old_fa_filename = os.path.join(extracted_dir, "OpenSubtitles.en-fa.fa")
    new_fa_filename = os.path.join(extracted_dir, "OpenSubtitles.en-fa.fa.txt")

    if os.path.exists(old_en_filename):
        os.rename(old_en_filename, new_en_filename)
        print(f"Renamed '{old_en_filename}' to '{new_en_filename}'")
    else:
        print(f"Warning: '{old_en_filename}' not found. Skipping rename.")

    if os.path.exists(old_fa_filename):
        os.rename(old_fa_filename, new_fa_filename)
        print(f"Renamed '{old_fa_filename}' to '{new_fa_filename}'")
    else:
         print(f"Warning: '{old_fa_filename}' not found. Skipping rename.")

    with open(os.path.join(extracted_dir, "OpenSubtitles.en-fa.en.txt"), 'r') as infile, open(os.path.join(extracted_dir, 'english_partly.txt'), 'w') as outfile:
        first_lines = islice(infile, number_of_samples_for_train)
        outfile.writelines(first_lines)

    with open(os.path.join(extracted_dir, "OpenSubtitles.en-fa.fa.txt"), 'r') as infile, open(os.path.join(extracted_dir, 'farsi_partly.txt'), 'w') as outfile:
        first_lines = islice(infile, number_of_samples_for_train)
        outfile.writelines(first_lines)


class Normalize:
    """
    A transform class to normalize and clean English and Farsi texts.
    """
    def __init__(self, nlp_en_model, hazm_normalizer_obj):
        self.nlp_en = nlp_en_model
        self.hazm_normalizer = hazm_normalizer_obj

    def __call__(self, sample):
        en_text, fa_text = sample['en_text'], sample['fa_text']

        # --- English processing (lowercase + remove punctuation + tokenize with spaCy) ---
        en_text = en_text.lower()
        en_text = en_text.translate(str.maketrans('', '', string.punctuation))
        en_text = " ".join([tok.text for tok in self.nlp_en(en_text)])

        # --- Farsi processing (Hazm + regex cleanup) ---
        fa_text = self.hazm_normalizer.normalize(fa_text)
        fa_text = re.sub(r'[^\w\s\u0600-\u06FF]', '', fa_text)  # keep only Persian + spaces
        fa_text = re.sub(r'[–.؟٫‍٬،‍:‍؛‍‍]', '', fa_text)  # remove extra Persian punctuations
        fa_text = fa_text.strip()

        return {'en_text': en_text, 'fa_text': fa_text}


class PairedTextDataset(Dataset):
    def __init__(self, en_file_path, fa_file_path, tokenizer, transform, max_length=64):
        """
        Args:
            en_file_path (str): Path to the English text file.
            fa_file_path (str): Path to the Farsi text file.
            tokenizer: HuggingFace tokenizer.
            transform: Optional preprocessing (Normalize).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        # --- Load raw lines ---
        with open(en_file_path, 'r', encoding='utf-8') as f:
            self.en_raw_lines = f.readlines()
        with open(fa_file_path, 'r', encoding='utf-8') as f:
            self.fa_raw_lines = f.readlines()

        if len(self.en_raw_lines) != len(self.fa_raw_lines):
            raise ValueError("English and Farsi files must have the same number of lines")

        self.num_samples = len(self.en_raw_lines)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        en_text = self.en_raw_lines[idx].strip()
        fa_text = self.fa_raw_lines[idx].strip()

        sample = {'en_text': en_text, 'fa_text': fa_text}

        # --- Apply normalization if provided ---
        if self.transform:
            sample = self.transform(sample)

        # --- Add translation prompt for input ---
        input_prompt = "Translate this to Persian: " + sample['en_text']

        # --- Tokenize ---
        input_tokens = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        output_tokens = self.tokenizer(
            sample['fa_text'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "labels": output_tokens["input_ids"].squeeze(0),
        }


if __name__ == '__main__':
    pass
    