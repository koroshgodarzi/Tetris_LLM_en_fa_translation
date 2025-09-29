from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, AutoTokenizer
import torch
from dataloader import Normalize
import spacy
from hazm import Normalizer
import re
import os
from huggingface_hub import hf_hub_download
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dotenv import load_dotenv
# ---------------------
# Load Model and Tokenizer
# ---------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

# --- Hazm Normalizer ---
hazm_normalizer = Normalizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
def get_model():
    repo_id = "alizali/Translate_Tetris"
    filename = "final.pt" 

    local_dir = "./finetuned_model"
    if not os.path.exists('filename'):
        try:
            # Download the file
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=HF_TOKEN
            )
            print(f"Downloaded '{filename}' from '{repo_id}' to '{downloaded_file_path}'")

        except Exception as e:
            print(f"Error downloading file: {e}")
            print(f"Please check if the file '{filename}' exists in the repository '{repo_id}' on the Hugging Face Hub.")

    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100", token=HF_TOKEN)
    model = prepare_model_for_kbit_training(model)

    proj_candidates = set()
    pattern = re.compile(r'(?:^|\.)(q_proj|k_proj|v_proj|out_proj|o_proj|gate_proj|up_proj|down_proj)(?:$|\.)')

    for name, _ in model.named_modules():
        m = pattern.search(name)
        if m:
            proj_candidates.add(m.group(1))

    if len(proj_candidates) == 0:
        proj_candidates = {"q_proj", "k_proj", "v_proj", "out_proj"}

    target_modules = sorted(list(proj_candidates))
    print("Using LoRA target modules:", target_modules)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.load_state_dict(torch.load(os.path.join(local_dir , filename), map_location=torch.device(device)))
    return model


class Translator:
    def __init__(self, model, tokenizer, normalizer, device, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.max_length = max_length
        self.device = device
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def translate(self, text):
        normalized_text = self.normalizer({'en_text': text})['en_text']

        input_prompt = normalized_text

        input_tokens = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                max_length=self.max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

normalizer = Normalize(nlp_en, hazm_normalizer, inference=True)
tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100", tgt_lang="fa")
translator = Translator(get_model(), tokenizer, normalizer, device)


# ---------------------
# FastAPI App
# ---------------------
app = FastAPI(title="Tetris Translation API")

# Template for prompt
TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
CONTEXT = (
    "This is a conversation with Tetris. It is an artificial intelligence model"
    "designed by a team of NLP experts at the Rahnemacollege to help you with Translation English to Persian"
)

# Request/Response Models
class TranslationRequest(BaseModel):
    en_text: str
    max_tokens: int = 256

class TranslationResponse(BaseModel):
    output: str


# ---------------------
# Endpoint
# ---------------------
@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    
    result = translator.translate(req.en_text)

    return TranslationResponse(output=result)
