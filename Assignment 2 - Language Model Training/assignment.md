# Assignment 2: Language Model Training

## Goal

Prepare a small text corpus, build a character vocabulary, encode and decode text, split data for training and validation, train a bigram model, compare samples, and introduce self-attention conceptually.

## Dataset and preparation

The training corpus is a compact collection of original, illustrative name-like strings included in `data/names.txt`. It is deliberately small so the complete exercise runs quickly and can be inspected. One name is stored per line. A newline is used as the end-of-name token, allowing the model to learn when to stop.

## Vocabulary, encoding, and decoding

The notebook constructs a sorted vocabulary from all characters in the corpus, plus a dedicated start token. Each character maps to an integer ID. The `encode` function maps strings to IDs and `decode` maps IDs back to characters. A short round-trip check confirms the two operations agree.

## Train and validation split

Names are shuffled with a fixed random seed and split 80/20. This keeps the validation examples separate from the training examples and makes the split repeatable. The model only counts adjacent-character pairs from the training set.

## Bigram model and loss

For every character, the model estimates the probability of the next character from observed transitions. Add-one smoothing assigns a small nonzero probability to unseen transitions. Training and validation negative log-likelihood are calculated over their respective next-character pairs. This count-based bigram model is a small language model: it learns one-step character transitions without a neural network.

The notebook records loss before training (uniform next-character probabilities) and after fitting the transition counts. The generated samples are produced once from the untrained uniform baseline and once from the fitted model, using the same start token and seeded sampling for a fair, reproducible demonstration.

## Results

Run `training.ipynb` or `train.py` to regenerate `results/losses.json`, `results/loss_curve.png`, `results/samples.txt`, and `model/bigram_model.json`. The saved model contains the vocabulary and smoothed transition counts and can be loaded without retraining.

## Before and after comparison

The untrained baseline samples characters uniformly, so it tends to look like noise and has no learned stopping behavior. The trained model favors character pairs seen in the corpus, which makes its samples more name-like. Since this is a tiny one-character-context model, it can still make awkward transitions and does not understand meaning.

## Conceptual introduction to self-attention

A bigram model uses only the immediately preceding character to choose the next one. Self-attention lets each position compute a weighted combination of information from many earlier positions. In a language model, this gives the prediction access to a broader context, such as the start of a name or phrase, instead of only its last character. Attention is a mechanism used inside larger neural models; this assignment's count-based model does not implement it.

## Reproducibility

From this folder, run `python train.py`. The script uses only the Python standard library and writes the model and results into this project. The notebook contains the same workflow in an instructional, cell-by-cell form.
