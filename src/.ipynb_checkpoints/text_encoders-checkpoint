from transformers import CLIPTextModel, CLIPTokenizer
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer
from transformers import BertModel

class clip_model:
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPTextModel.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")

        # Encode the text
        outputs = self.model(**inputs)

        # The text embeddings are in outputs.last_hidden_state
        text_embeddings = outputs.last_hidden_state

        return text_embeddings


class bert_model:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_len = 128

    def embed_text(self, text):
        encoded_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=512,           # Pad/truncate to a maximum length (BERT's max is 512)
            padding='max_length',     # Pad with [PAD] tokens
            truncation=True,          # Truncate if longer than max_length
            return_attention_mask=True, # Return attention mask
            return_token_type_ids=True, # Return token type IDs (for sentence pair tasks)
            return_tensors='pt'       # Return PyTorch tensors
        )
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        token_type_ids = encoded_inputs['token_type_ids']

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # The last hidden state contains the contextual embeddings for each token
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states