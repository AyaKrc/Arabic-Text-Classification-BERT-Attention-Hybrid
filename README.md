> ⚠️ **PS:** If the `.ipynb` version doesn’t load properly, please check the PDF version instead. ⚠️


# Arabic-Text-Classification-BERT-Attention-Hybrid 

## 📄 Overview

This project is an **Advanced Natural Language Processing (NLP)** undertaking focused on **Arabic Text Classification**. It involves **fine-tuning** a state-of-the-art pre-trained language model, **BERT-Base-ArabertV2**, and experimentally enhancing its architecture by integrating a novel **Multi-Head Differential Attention** mechanism.

The core goal is to explore how a hybrid attention model performs compared to the standard BERT model on a large-scale Arabic news dataset.

---

## 🎯 Project Objectives

The project was executed with the following three primary objectives:

1. **Fine-Tuning the Target Model**
   - Fine-tune the pre-trained **BERT-Base-ArabertV2** model on the **SANAD Dataset** for the Arabic Text Classification task.  
   - Evaluate the model's performance rigorously using standard metrics from **TorchMetrics**.

2. **Implementing Multi-Head Differential Attention**
   - Develop a custom PyTorch class to implement the **Multi-Head Differential Attention** mechanism.  
   - Integrate this custom class into the fine-tuned BERT architecture.

3. **Experimenting with Hybrid Attention Mechanisms**
   - Replace a strategic percentage (**25% to 50%**) of the original **Multi-Head Self-Attention** layers within the **encoder** and/or **decoder** with the new differential attention mechanism.  
   - Train and assess the performance of the modified hybrid models for each configuration to determine the impact of the experimental attention layer.

---

## 📚 Dataset and Task

| Feature | Details |
| :--- | :--- |
| **Task** | **Arabic Text Classification** (Multi-class) |
| **Dataset Name** | **SANAD Dataset (Subset)** |
| **Labels** | **7 Unique Categories:** Religion, Tech, Sports, Culture, Politics, Finance, and Medical |
| **Training Size** | **99,810** samples |
| **Testing Size** | **11,090** samples |
| **Preprocessing** | Text tokenization using the `aubmindlab/bert-base-arabertv2` tokenizer with a maximum sequence length of **512 tokens** |

---

## 🧠 Model and Architecture

### Base Model

- **Model:** **BERT** (Bidirectional Encoder Representations from Transformers)  
- **Pre-trained Version:** **BERT-Base-ArabertV2** (`aubmindlab/bert-base-arabertv2`)  
- **Role:** The base model provides robust Arabic language understanding features, which are then adapted (fine-tuned) for the classification task.

### Custom Component: Multi-Head Differential Attention

The standard **Multi-Head Self-Attention** layers are replaced with a custom implementation of **Multi-Head Differential Attention**.  
This modification is designed to explore new ways of calculating attention scores and managing context, potentially offering performance gains or novel insights into model dynamics.

### Hybrid Configuration

The project implements and tests various hybrid models where the custom differential attention replaces a portion of the standard attention layers.  
The primary configurations tested involve swapping **25%** and **50%** of the attention heads in the base model's layers.

---

## 💻 Installation and Setup

### Prerequisites

- Python (3.x)  
- CUDA-enabled GPU (recommended for training BERT)

