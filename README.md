# WEX 428 — Handwritten Math Symbol AI

Full-stack ML platform for handwritten mathematical symbol recognition, with
self-supervised learning and domain generalization comparisons.

## Repo structure

```
wex428-project/
├── backend/            # FastAPI (or your chosen framework) — auth, prediction API, stats
├── frontend/           # React / HTML+JS — landing, auth, dashboard, symbol recognition UI
├── ml/
│   ├── data/           # dataset loading utilities
│   ├── models/         # saved model weights (.pt/.h5) — one subfolder per approach
│   ├── notebooks/       # exploratory / experimental notebooks
│   ├── scripts/        # reusable scripts (eda.py, train.py, evaluate.py, etc.)
│   └── requirements.txt
├── data/
│   ├── raw/            # put the downloaded Kaggle dataset here, unzipped
│   └── processed/      # resized/cleaned images, split indices, etc.
└── docs/               # report drafts, EDA output, diagrams
```

## Getting started (Phase 1)

1. Download the dataset from the Kaggle link / Google Drive link provided in
   the course PDF, unzip it, and place its contents into `data/raw/`.
   - If the dataset is class-per-folder (e.g. `data/raw/0/`, `data/raw/+/`, ...),
     no changes needed.
   - If it's a flat folder + CSV of labels, or pixel-in-CSV format, tell me and
     I'll adjust `ml/scripts/eda.py`'s `load_image_dataset()` function.

2. Set up a Python environment for the ML side:
   ```bash
   cd ml
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the EDA script:
   ```bash
   cd ml/scripts
   python eda.py --data_dir ../../data/raw --out_dir ../../docs/eda_output
   ```
   This will print class distribution, image size/mode stats, a proposed
   train/val/test split, and save a sample grid image + class distribution
   chart into `docs/eda_output/`.

4. Review `docs/eda_output/class_distribution.png` and `sample_grid.png` —
   check whether classes are balanced and whether images need resizing.

## Next phases

- **Phase 2**: baseline CNN/ResNet training script (`ml/scripts/train_baseline.py`)
- **Phase 3**: augmentation-based training variant
- **Phase 4**: self-supervised pretraining + domain generalization
- **Phase 5**: FastAPI backend
- **Phase 6**: frontend
- **Phase 7**: formula segmentation + reconstruction
- **Phase 8**: Docker Compose deployment

Come back once Phase 1 is done (or if the dataset layout differs from what's
assumed here) and we'll build the Phase 2 training script next.
