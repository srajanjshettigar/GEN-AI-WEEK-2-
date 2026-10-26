# Mini Project: Character-Level Text Generator

Train a small character-level bigram model that generates name-like text from a starter character or sequence. The project includes an instructional notebook, a reproducible 80/20 data split, before/after loss metrics, a saved JSON model, and ten samples continued from the starter `a`.

## Run

From this folder, run `python train.py` to regenerate the model and results. Or open `training.ipynb` in Jupyter and run each cell from top to bottom. The project uses only the Python standard library.

## Outputs

- `data/names.txt`: small illustrative training corpus.
- `model/bigram_model.json`: saved vocabulary and smoothed transition probabilities.
- `results/losses.json` and `results/losses.csv`: uniform-baseline and trained train/validation negative log-likelihood.
- `results/samples.txt`: ten generated samples, a before-training comparison, and ten outputs continued from starter `a`.

This intentionally small model looks at one previous character. It demonstrates next-character prediction; it is not a general-purpose language model.
