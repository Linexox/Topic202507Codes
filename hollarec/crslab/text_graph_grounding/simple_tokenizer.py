from transformers import AutoTokenizer
model_path = 'D:\\.Workspace\\.MODEL\\HF-Model-Backup\\llava-1.5-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)

class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(self, text: str, max_length: int = 77):
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)