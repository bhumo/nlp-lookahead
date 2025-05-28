## 📊 Results

We evaluated three dynamic variants of the Lookahead Optimizer — **Adaptive Decrease K**, **Adaptive Increase K**, and **Adaptive α Lookahead** — across vision and language modeling tasks. The experiments were conducted on the **CIFAR-10** dataset for image classification and the **Penn Treebank** dataset using **Transformer** and **LSTM** models for language modeling.

---

### 🖼️ CIFAR-10 (Image Classification)

| Optimizer                | Test Accuracy (%)     |
|-------------------------|-----------------------|
| SGD (LR=0.1)            | 96.63 ± 0.10          |
| Lookahead (K=10, α=0.5) | 96.61 ± 0.03          |
| Adaptive Decrease K     | 96.60 ± 0.09          |
| **Adaptive Increase K** | **96.72 ± 0.17**      |
| Adaptive α Lookahead    | 96.72 ± 0.12          |


To see the code & results of Lookahead on CIFAR-10: https://github.com/Jayant1234/genLoss/tree/bhumika/glam_vision_datasets/example/greedy_lookahead_optimizer

This folder contains all the experiments conducted on Adavance Lookahead Optimizer such as Adaptive Decrease K, Adaptive Increase K, Adaptive Alpha Lookahead Optimizer.

✅ **Adaptive Increase K** showed the best performance on CIFAR-10, demonstrating robust generalization and stability under changing learning rates.

---

### 📚 Penn Treebank (Language Modeling – Transformer)

| Optimizer                | Test Perplexity (↓)   |
|-------------------------|-----------------------|
| ADAM                    | **91.98 ± 0.15**      |
| Lookahead (K=10)        | 94.64 ± 0.21          |
| Adaptive Decrease K     | 94.24 ± 0.05          |
| **Adaptive Increase K** | **91.92 ± 0.23**      |
| Adaptive α Lookahead    | 93.54 ± 0.04          |

⚖️ **Adaptive Increase K** slightly outperformed ADAM, while both adaptive methods improved over static Lookahead.

---

### 🧠 Penn Treebank (Language Modeling – LSTM)

| Optimizer                | Test Perplexity (↓)   |
|-------------------------|-----------------------|
| ADAM                    | **125.02 ± 3.48**     |
| Lookahead (K=10)        | 131.32 ± 1.48         |
| Adaptive Decrease K     | 132.84 ± 0.27         |
| **Adaptive Increase K** | **124.98 ± 1.87**     |
| Adaptive α Lookahead    | **125.00 ± 0.84**     |

📉 **Adaptive Increase K** and **Adaptive α Lookahead** closely matched or slightly outperformed ADAM, significantly improving over static Lookahead.

---
### Experiments Analysis 
The analysis of the results found are explained in the below project report

Report Link: https://outlookuga-my.sharepoint.com/personal/bg91882_uga_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbg91882%5Fuga%5Fedu%2FDocuments%2FAttachments%2FAdaptive%20Lookahead%2FMasters%5Freport%5Fadaptive%5Flookahead%2Epdf&parent=%2Fpersonal%2Fbg91882%5Fuga%5Fedu%2FDocuments%2FAttachments%2FAdaptive%20Lookahead&ga=1
### 🔍 Summary

- **CIFAR-10**: Adaptive Increase K achieved the highest accuracy and offered better performance than static Lookahead and SGD baselines.
- **Transformer & LSTM**: Adaptive α and Increase K variants performed on par or better than ADAM in terms of test perplexity.
- **Adaptive α Lookahead** showed potential for performance tuning through validation-driven interpolation adjustments.

> 💡 These results suggest that dynamically tuning `K` and `α` in Lookahead optimizers can enhance model robustness and adaptability, depending on the task and architecture.

# Contribution:
Quicker Repo is used for the base setup

The following changes have been made to the base repo:
Code for Training Adaptive Alpha Lookahead optimizer on NLP Tasks: nlp-lookahead/Quixer/quixer/setup_training_adaptive_alpha.py

The code is to be run in general since Adaptive Alpha Lookahead Optimizer is just a optimizer it can be easily replaced by any of the other methods


