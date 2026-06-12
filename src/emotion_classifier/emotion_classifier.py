import os
import pickle
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from transformers import BertTokenizer
from src.emotion_classifier.model import EmotionClassifierModel as BiLSTMNet

EMOTION_LABELS = ['sadness','joy','love','anger','fear','surprise']

ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_DIR = _ROOT / "models" / "emotion_classifier"

class EmotionClassifier:

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None) -> None:
        self._dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self._model = None
        self._tokenizer = None
        self._max_len = 128
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        from transformers import BertTokenizerFast

        cfg_path = self._dir / "emotion_config.yaml"
        weights_path = self._dir / "emotion_classifier_best_model.pt"
        tok_path = self._dir / "tokenizer.pkl"

        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Required model files missing in {self._dir}.\n"
                f"Ensure 'emotion_config.yaml' and 'emotion_classifier_best_model.pt' are present."
            )

        print(f"Reading configuration from {cfg_path}...")
        with open(cfg_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        def get_val(key: str, default):
            item = yaml_data.get(key)
            if isinstance(item, dict) and "value" in item:
                return item["value"]
            return default

        config_data = {
            "embed_dim":     int(get_val("embed_dim", 128)),
            "lstm_units":    int(get_val("lstm_units", 256)),
            "num_layers":    int(get_val("num_layers", 2)),       # Defaulting to 2 layers
            "dropout":       float(get_val("dropout", 0.25)),
            "fc_hidden_dim": int(get_val("fc_hidden_dim", 64)),   # Defaulting to 64 hidden units
            "num_classes":   int(get_val("num_classes", 6)),      # 6 core target emotions
            "vocab_size":    int(get_val("vocab_size", 30522)),   # bert-base-uncased matrix limit
        }
        
        self._max_len = int(get_val("max_len", 45))

        print(f" Initializing Tokenizer (bert-base-uncased)...")
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        print(f"🧠 Building BiLSTM and injecting state dict...")
        self._model = BiLSTMNet(config=config_data)
        
        checkpoint = torch.load(weights_path, map_location=self._device)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            print(" Training checkpoint wrapper detected. Extracting 'model_state' weights...")
            real_state_dict = checkpoint["model_state"]
        else:
            real_state_dict = checkpoint

        self._model.load_state_dict(real_state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._loaded = True
        print("Ok BiLSTM Emotion Classifier state weights successfully mapped and loaded!")

    def predict(self, text: str) -> dict:
        """Alias to keep consistent with the FastAPI endpoint expectations."""
        return self.classify(text)

    def classify(self, text: str) -> dict:
        self._load()

        if not isinstance(text, str) or not text.strip():
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

        encoded_inputs = self._tokenizer(
            text,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_inputs["input_ids"].to(self._device)
        attention_mask = encoded_inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        all_scores = {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)}
        top_emotion = max(all_scores, key=all_scores.__getitem__)

        return {
            "emotion": top_emotion,
            "confidence": all_scores[top_emotion],
            "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1]))
        }