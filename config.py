# used in tokenizer.py

MAX_LEN        = 50
BATCH_SIZE     = 64
TOKENIZER_PATH = 'models/emotion_classifier/tokenizer.pkl'


# used in model.py

VOCAB_SIZE  = 30522   # bert-base-uncased vocab size
EMBED_DIM   = 128
LSTM_UNITS  = 128
DROPOUT     = 0.4
NUM_CLASSES = 6


# Wandb
WANDB_PROJECT = "mental-health-chatbot"
WANDB_RUN_NAME = "bilstm-emotion-classifier"