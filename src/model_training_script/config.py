LABELS = ["섹시", "응원", "입덕", "퍼포먼스", "유머", "재방문", "기타"]
NUM_LABELS = len(LABELS)

MODEL_NAME = "klue/roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
THRESHOLD = 0.5

LABELED_PATH = "../data/comments_labeled.jsonl"
TRAIN_PATH = "../data/train.jsonl"
VAL_PATH = "../data/val.jsonl"
SAVE_PATH = "../model_output"
