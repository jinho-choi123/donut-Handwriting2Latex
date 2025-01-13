
config = {
        "INK_SEQ_MIN": -500,
        "INK_SEQ_MAX": 500,
        "INK_TIME_SAMPLING_DELTA": 30,
        "INK_PADDING": 4,
        "IMG_SIZE": 224,

        "PRETRAINED_REPO_ID": "naver-clova-ix/donut-base",
        "FINTUNED_REPO_ID": "ball1433/Handwriting2Latex",

        "max_length": 128,
        "max_new_tokens": 100,
        "INIT_LR": 1e-4,
        "batch_size": 16,
        "num_workers": 4,
        "lora_r": 8,
        "load_lora": False,

        "wandb_project": "Handwriting2Latex",

        }
