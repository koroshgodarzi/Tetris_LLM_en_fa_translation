# English-Farsi Subtitle Translation with LoRA Fine-Tuning

A PyTorch-based project for fine-tuning a multilingual translation model (`M2M100ForConditionalGeneration`) using **LoRA (Low-Rank Adaptation)** on English-Farsi subtitle pairs from OpenSubtitles. The project includes data preprocessing, model training, a FastAPI-based translation API, and Docker deployment.

---

## Features

* Fine-tunes M2M100 for parameter-efficient English-to-Farsi translation.
* Preprocesses English text with SpaCy and Farsi text with Hazm.
* Dynamic dataset downloading and cleaning.
* Tracks validation loss and saves the best model automatically.
* Provides a FastAPI endpoint for real-time translation.
* Supports containerized deployment with Docker.

---

## Table of Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Training](#training)
* [API Usage](#api-usage)
* [Docker Deployment](#docker-deployment)
* [Project Structure](#project-structure)
* [Training Details](#training-details)
* [Contributing](#contributing)
* [License](#license)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/english-farsi-translation.git
cd english-farsi-translation
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Dataset

This project uses English-Farsi subtitle pairs from **OpenSubtitles**.

* `dataloader.py` contains the `download_opus_subtitles` function which downloads and extracts the dataset.
* Only a subset of lines is used for training (`english_partly.txt` and `farsi_partly.txt`).
* English text is normalized with SpaCy, and Farsi text is normalized and cleaned with Hazm.

---

## Training

Run the training script:

```bash
python train.py
```

The script will:

* Download and preprocess the dataset (if not already present).
* Split the dataset into 90% training and 10% validation.
* Fine-tune the M2M100 model using LoRA.
* Save the best model weights to `./model_weights/`.

### Example: Load the Model for Inference

```python
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100", tgt_lang="fa")
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
model.load_state_dict(torch.load("model_weights/best_model_epoch_X.pt"))
model.eval()

text = "Translate this to Persian: How are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## API Usage (FastAPI)

The project includes a FastAPI application (`api.py`) for real-time translation.

### Run API locally:

```bash
uvicorn api:app --reload
```

### Endpoint

* **POST** `/translate`

Request body:

```json
{
  "en_text": "Hello, how are you?",
  "max_tokens": 256
}
```

Response:

```json
{
  "output": "سلام، حال شما چطور است؟"
}
```

---

## Docker Deployment

Run the FastAPI translation API in a Docker container:

1. **Build the Docker image**:

```bash
docker build -t en-fa-translation .
```

2. **Run the container**:

```bash
docker run -d -p 8000:8000 en-fa-translation
```

3. API will be available at:

```
http://localhost:8000/translate
```

4. Test with `curl`:

```bash
curl -X POST "http://localhost:8000/translate" \
-H "Content-Type: application/json" \
-d '{"en_text": "Hello, how are you?", "max_tokens": 256}'
```

---

## Project Structure

```
english-farsi-translation/
│
├── dataloader.py           # Dataset and preprocessing utilities
├── train.py                # Main training script
├── api.py                  # FastAPI service for inference
├── requirements.txt        # Python dependencies
├── english_partly.txt      # Subset of English subtitles
├── farsi_partly.txt        # Subset of Farsi subtitles
├── model_weights/          # Saved model weights
├── Dockerfile              # Dockerfile for containerized deployment
└── README.md               # This file
```

---

## Training Details

* **Model:** `M2M100ForConditionalGeneration` (pretrained `small100`)
* **LoRA Parameters:**

  * Rank `r=8`
  * Alpha `32`
  * Dropout `0.05`
  * Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj` (auto-detected)
* **Optimizer:** `AdamW` with learning rate `5e-5`
* **Scheduler:** Linear with 100 warmup steps
* **Epochs:** 10
* **Batch Size:** 32

Validation loss is tracked, and the best model is automatically saved.

---

## Contributing

Contributions are welcome! You can:

* Open issues for bug reports or feature requests
* Submit pull requests for improvements

---

## License

This project is licensed under the **MIT License**.
