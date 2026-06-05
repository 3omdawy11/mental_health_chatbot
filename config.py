import os

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input")

if IS_KAGGLE:
    WORKING_DIR = "/kaggle/working/project/"
else:
    WORKING_DIR = ""

EMOTION_SPLIT_PATH = WORKING_DIR + 'data/processed/emotion_{}.csv'  # expects 'train', 'val', 'test' in {}
MODEL_SAVE_PATH = WORKING_DIR + 'models/emotion_classifier/emotion_classifier_best_model.pt'


#tokenizer
MAX_LEN        = 50
BATCH_SIZE     = 64
TOKENIZER_PATH = WORKING_DIR + 'models/emotion_classifier/tokenizer.pkl'




# Wandb
WANDB_PROJECT = "mental-health-chatbot"




# Hyperparameters for training for each run

BASELINE_RUN_CONFIG = {
    "name"          : "baseline-bilstm",
    "epochs"        : 10,
    "learning_rate" : 1e-3,
    "batch_size"    : 64,
    "embed_dim"     : 128,
    "lstm_units"    : 128,
    "num_layers"    : 2,
    "dropout"       : 0.4,
    "fc_hidden_dim" : 64,
    "max_len"       : 50,
    "num_classes"   : 6,      
    "vocab_size"    : 30522, 
}

RUN2_CONFIG = {**BASELINE_RUN_CONFIG, "name": "lr-tuning",   "learning_rate": 3e-4, "epochs": 30}
RUN3_CONFIG = {**RUN2_CONFIG,         "name": "bigger-lstm",  "lstm_units": 256, "epochs": 30}
RUN4_CONFIG = {**RUN3_CONFIG,         "name": "regularized",  "dropout": 0.5, "epochs": 30}

RUN5_CONFIG = {
    "name": "ablation-high-capacity-regularized",
    "epochs": 25,
    "learning_rate": 3e-4,        
    "lstm_units": 512,
    "num_layers": 3,
    "embed_dim": 256,
    "fc_hidden_dim": 128,
    "dropout": 0.35,               
    "batch_size": 64,           
    "max_len": 64,
    "num_classes"   : 6,       
    "vocab_size"    : 30522,
}

RUN6_CONFIG = {
    "name": "ablation-optimized-context",
    "epochs": 20,
    "learning_rate": 1e-3,        
    "lstm_units": 256,              # Cut capacity in half to stop memorization
    "num_layers": 2,                
    "embed_dim": 128,
    "fc_hidden_dim": 64,
    "dropout": 0.25,
    "batch_size": 32,
    "max_len": 45,     
    "num_classes"   : 6,      
    "vocab_size"    : 30522,  
}
