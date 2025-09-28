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

# ---------------------
# Load Model and Tokenizer
# ---------------------
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
    filename = "final.pt" # Assuming the .pt file is named best_model.pt

    # Define the local path where you want to save the downloaded file
    # '.' means the current directory
    local_dir = "."
    try:
        # Download the file
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False # Set to False to avoid symlinks
        )
        print(f"Downloaded '{filename}' from '{repo_id}' to '{downloaded_file_path}'")

    except Exception as e:
        print(f"Error downloading file: {e}")
        print(f"Please check if the file '{filename}' exists in the repository '{repo_id}' on the Hugging Face Hub.")

    

        model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
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
        model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
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
        # Normalize the input text
        # Assuming the normalizer expects a dictionary with 'en_text' key
        normalized_text = self.normalizer({'en_text': text})['en_text']

        # Add the translation prompt
        input_prompt = normalized_text

        # Tokenize the input
        input_tokens = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                max_length=self.max_length,
                num_beams=5, # Example: use beam search
                early_stopping=True,
                no_repeat_ngram_size=2 # Example: prevent repeating ngrams
            )

        # Decode the output tokens
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        '''
        # Post-process the translated text if necessary (e.g., remove the prompt)
        # This depends on how your model was trained. If it was trained to output
        # only the Farsi translation after the prompt, you might need to split the string.
        # For a causal LM trained with input+output, the output will likely include the input and prompt.
        # You might need to find the start of the translated text.

        # A simple approach assuming the model outputs "Translate this to Persian: [English Text] [Farsi Text]"
        # Find the end of the English text and prompt
        prompt_end_index = translated_text.find(normalized_text) + len(normalized_text)
        # Assuming the Farsi translation starts right after the English text and prompt
        # This might need adjustment based on actual model output format
        if prompt_end_index != -1:
             # Find the start of the Farsi part after the prompt and English text
             # This is a heuristic and might need refinement
             farsi_start_index = translated_text[prompt_end_index:].find(":") # Example heuristic, look for a separator
             if farsi_start_index != -1:
                 translated_text = translated_text[prompt_end_index + farsi_start_index + 1:].strip()
             else:
                # If no separator found, assume the Farsi starts immediately after the English text
                translated_text = translated_text[prompt_end_index:].strip()
        '''

        return translated_text

normalizer = Normalize(nlp_en, hazm_normalizer)
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
    text: str
    max_tokens: int = 256

class TranslationResponse(BaseModel):
    output: str


# ---------------------
# Endpoint
# ---------------------
@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    
    result = translator.translate(req.text)
    result = output_text[len(model_input):]

    return TranslationResponse(output=result)
