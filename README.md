# Smart Spell Checker

A French spelling-correction project developed as part of my Master's studies in Intelligent Systems Engineering.

## Overview

This project explores Natural Language Processing approaches for detecting and correcting French spelling errors. It includes a supervised Seq2Seq LSTM autoencoder trained using the Lexique383 French lexicon and compares that approach with language-model-based correction strategies.

## Key Work

- Built a French spelling-correction workflow in Python.
- Trained a Seq2Seq LSTM autoencoder for spelling correction.
- Used the Lexique383 lexical resource for French vocabulary data.
- Compared supervised neural correction with language-model-based strategies.

## Try the French spelling demo

From the repository root, run `python3 spell_demo.py bonjor` to print `bonjour`. Run `python3 spell_demo.py --evaluate` to reproduce the table and inspect every error. Python 3.9+ is sufficient.

This small companion model learns substitution costs from 17 hand-authored French spelling pairs and ranks a fixed word list by weighted edit distance. The baseline uses ordinary Levenshtein distance with the same vocabulary and tie-breaking rule. The 16 illustrative test cases use target words absent from training but present in the candidate vocabulary.

| Decoder | Exact matches |
| --- | --- |
| Learned substitution costs | 13/16 (81.25%) |
| Levenshtein | 12/16 (75.00%) |

For `etage`, the learned accent cost selects `étage` while the baseline returns `stage`. Both retain `cour`, `marie`, and `porte` when the intended labels are `cours`, `marié`, and `porté`. Each input is already a valid French word, illustrating the ambiguity of isolated-word correction. This hand-authored sample measures only the companion demo.

## Original neural project status

This repository publishes the Seq2Seq LSTM architecture and configuration. It does not yet include the training pipeline, model weights, or a reproducible inference example. Held-out metrics and baseline results are pending publication.

## Technologies

- Python
- Natural Language Processing (NLP)
- Deep Learning
- Seq2Seq
- LSTM

## Context

Master's project — Intelligent Systems Engineering, Ibn Tofail University.

## Author

**Mohamed Laaribi**  
GitHub: @MO7KUN
