import os

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input")

if IS_KAGGLE:
    WORKING_DIR = "/kaggle/working/project/"
else:
    WORKING_DIR = ""

# locations
EMOTION_SPLIT_PATH = WORKING_DIR + 'data/processed/emotion_{}.csv'  # expects 'train', 'val', 'test' in {}
MODEL_SAVE_PATH = WORKING_DIR + 'models/emotion_classifier/emotion_classifier_best_model.pt'



# used in tokenizer.py

MAX_LEN        = 50
BATCH_SIZE     = 64
TOKENIZER_PATH = WORKING_DIR + 'models/emotion_classifier/tokenizer.pkl'




# Wandb
WANDB_PROJECT = "mental-health-chatbot"




# Hyperparameters for training for each run

# config.py

BASELINE_RUN_CONFIG = {
    "name"          : "baseline-bilstm",
    "epochs"        : 20,
    "learning_rate" : 1e-3,
    "batch_size"    : 64,
    "embed_dim"     : 128,
    "lstm_units"    : 128,
    "num_layers"    : 2,
    "dropout"       : 0.4,
    "fc_hidden_dim" : 64,
    "max_len"       : 50,
    "num_classes"   : 6,       # ← added
    "vocab_size"    : 30522,   # ← added (bert-base-uncased)
}

RUN2_CONFIG = {**BASELINE_RUN_CONFIG, "name": "lr-tuning",   "learning_rate": 3e-4}
RUN3_CONFIG = {**RUN2_CONFIG,         "name": "bigger-lstm",  "lstm_units": 256}
RUN4_CONFIG = {**RUN3_CONFIG,         "name": "regularized",  "dropout": 0.5}