import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from .config import MODEL_NAME, MAX_LENGTH

class Embedder:
    def __init__(self, model_name=MODEL_NAME, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Loading {model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def encode(self, texts, batch_size=32, centre=True):
        """
        Produce sentence embeddings for a list of texts.

        ``centre=True`` subtracts the corpus mean before L2-normalisation to
        correct BERT-style anisotropy (representations collapsed into a cone).
        """
        raw_chunks = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            raw_chunks.append(batch_embeddings.cpu().numpy())

        raw = np.vstack(raw_chunks)
        if centre:
            raw = raw - raw.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / np.clip(norms, 1e-9, None)).astype(np.float32)
