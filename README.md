# Autoregressive Text Generation System (ICT4403 Assignment 2)

An autoregressive language modeling system built to evaluate sequential text-generation capabilities across four recurrent neural network architectures: **Simple RNN**, **Bidirectional RNN**, **LSTM**, and **GRU**.

## Problem Statement and Objective
Autoregressive language models predict the conditional probability distribution of the next token $x_t$ given prior context tokens $x_1, \dots, x_{t-1}$. The objective of this assignment is to implement, evaluate, and benchmark Simple RNN, BiRNN, LSTM, and GRU under standardized training conditions on a classic literary corpus, evaluating both formal statistical metrics (loss, perplexity, accuracy) and generation behaviors across different sampling temperatures.

## Dataset Description and Source
* **Corpus**: *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle
* **Source**: [Project Gutenberg EBook #1661](https://www.gutenberg.org/ebooks/1661)
* **Characteristics**: Narrative English containing ~126,000 cleaned tokens. Stripped of Gutenberg header/footer license text, lowercased, and tokenized with syntax preserved. The vocabulary is capped at the top 7,000 tokens with an `<unk>` token handling low-frequency vocabulary.
* **Sequences**: Constructed with a sliding window of 20 words predicting the 21st word, split chronologically into 80% Train, 10% Validation, and 10% Test sets to prevent overlapping sequence data leakage.

## Model Architectures Implemented
All models share identical baselines: **Embedding Dimension = 128**, **Hidden Dimension = 256**, **Sequence Length = 20**, and **Dropout = 0.20**.
1. **Simple RNN**: Standard recurrent network; vulnerable to vanishing gradients across the 20-token horizon.
2. **Bidirectional RNN**: Encodes the 20-token historical window in both forward and backward directions ($2 \times 256$ hidden units), concatenating the terminal hidden states to predict the next word.
3. **LSTM**: Gated memory cells with input, forget, and output gates mitigating vanishing gradients.
4. **GRU**: Compact gating mechanism combining cell and hidden states using reset and update gates.

## Execution Steps and Required Dependencies
```bash
# Clone the repository
git clone [https://github.com/aryan-nair/Sherlock-GenText-Autoregressive-LM.git](https://github.com/aryan-nair/Sherlock-GenText-Autoregressive-LM.git)
cd Sherlock-GenText-Autoregressive-LM

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook autoregressive_text_generation.ipynb
