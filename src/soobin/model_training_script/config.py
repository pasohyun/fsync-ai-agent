from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

LABELS = [
    "기타_노이즈",
    "비주얼_멤버매력",
    "장기_팬덤",
    "역주행_기대",
    "보컬_라이브",
    "신규_유입",
    "밴드_정체성",
    "위로_공감",
    "연주_악기",
    "이별_감성",
    "청량_여름",
    "음악성",
]
NUM_LABELS = len(LABELS)

MODEL_NAME = "klue/roberta-large"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
THRESHOLD = 0.5

LABELED_PATH = str(PROJECT_ROOT / "corpus_labeled_1.jsonl")
TRAIN_PATH = str(BASE_DIR / "train.jsonl")
VAL_PATH = str(BASE_DIR / "val.jsonl")
SAVE_PATH = str(PROJECT_ROOT / "model_output")
