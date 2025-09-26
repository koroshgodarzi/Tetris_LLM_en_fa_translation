from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------
# Load Model and Tokenizer
# ---------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./model"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ---------------------
# FastAPI App
# ---------------------
app = FastAPI(title="PersianMind Translation API")

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
    # Build prompt
    model_input = TEMPLATE.format(context=CONTEXT, prompt=req.text)

    # Tokenize
    input_tokens = tokenizer(model_input, return_tensors="pt").to(device)

    # Generate
    generate_ids = model.generate(
        **input_tokens,
        max_new_tokens=req.max_tokens,
        do_sample=False,
        repetition_penalty=1.1
    )

    # Decode
    output_text = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    result = output_text[len(model_input):]

    return TranslationResponse(output=result)
