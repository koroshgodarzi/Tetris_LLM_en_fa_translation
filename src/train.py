from torch.utils.data import random_split, DataLoader
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
import spacy
import hazm
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.optim import AdamW
import torch.nn.utils as nn_utils
from tqdm import tqdm
import os
from dataloader import PairedTextDataset, Normalize, download_opus_subtitles
# import bitsandbytes


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 10
    output = os.path.join('.', 'model_weights')
    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(os.path.join('english_partly.txt')):
        download_opus_subtitles()

    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm")

    hazm_normalizer = hazm.Normalizer()

    tokenizer = AutoTokenizer.from_pretrained("universitytehran/PersianMind-v1.0")

    dataset = PairedTextDataset(
        os.path.join('english_partly.txt'),
        os.path.join('farsi_partly.txt'),
        tokenizer,
        transform=Normalize(),
        max_length=64,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model = AutoModelForCausalLM.from_pretrained(
        "universitytehran/PersianMind-v1.0",
        device_map="auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True
    )

    # Step 1: Prepare model for k-bit training (adds gradient checkpointing, casts norms to fp32, etc.)
    model = prepare_model_for_kbit_training(model)

    # Step 2: Configure LoRA
    lora_config = LoraConfig(
        r=8,  # rank of LoRA matrices
        lora_alpha=32,  # scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} | Avg Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path} with val_loss={best_val_loss:.4f}")
