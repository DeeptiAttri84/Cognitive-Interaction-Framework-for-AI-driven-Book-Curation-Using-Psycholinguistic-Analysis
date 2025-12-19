<div align="center">

# 🧠 Feel2Read

### *Cognitive Interaction Framework for AI-Driven Book Curation*

**Emotion-Aware Book Recommendations Using Psycholinguistic Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Status](https://img.shields.io/badge/Project-Academic-success.svg)]()

[Features](#-key-features) • [Architecture](#️-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 📖 Overview

**Feel2Read** is an intelligent book recommendation system that goes beyond traditional approaches by understanding and responding to your **emotional and psychological state** in real-time. Using advanced psycholinguistic analysis and transformer-based AI models, it curates personalized reading suggestions that align with how you're feeling right now.

Unlike conventional recommendation engines that rely solely on past behavior or ratings, Feel2Read analyzes the emotional nuances in your input—whether text, voice, or image—to suggest books that resonate with your current mood and cognitive state.

### Why Feel2Read?

**Traditional systems** focus on *what* you've read.  
**Feel2Read** understands *how* you feel.

---

## 🎯 Problem Statement

Readers face several challenges when searching for their next book:

- **Emotional disconnect** → Existing systems don't consider current feelings or mental state
- **Generic recommendations** → One-size-fits-all suggestions ignore psychological needs
- **Limited wellness focus** → Mental-health-oriented reading guidance is scarce

### Our Solution

An intelligent framework that interprets emotional cues from multiple input types and delivers cognitively aligned book recommendations that support your emotional well-being.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎭 Multimodal Input

- 📝 **Text Analysis** — Express your mood or describe what you're looking for
- 🎙️ **Voice Recognition** — Speak naturally, we'll understand your tone
- 🖼️ **Image Processing** — Extract text from images using OCR

</td>
<td width="50%">

### 🧬 Emotion Intelligence

- **Psycholinguistic extraction** of cognitive patterns
- **Transformer-based** emotion detection (RoBERTa)
- **Smart emotion-to-genre** mapping algorithm
- Real-time sentiment analysis

</td>
</tr>
<tr>
<td width="50%">

### 📚 Smart Curation

- Genre classification with deep learning
- Context-aware ranking system
- Personalized recommendation logic
- Top-N book suggestions

</td>
<td width="50%">

### 💡 User Experience

- Clean, intuitive Streamlit interface
- Simple user profile management
- Fast response times
- Interactive dashboard

</td>
</tr>
</table>

---

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Emotion Detection** | Transformer (RoBERTa) | Identifies user's emotional state |
| **Genre Classification** | RoBERTa Embeddings | Categorizes books by genre |
| **Psycholinguistic Analyzer** | NLP Pipeline | Extracts cognitive features |
| **Recommendation Engine** | Hybrid Algorithm | Maps emotions to suitable books |
| **Frontend** | Streamlit | Interactive user interface |

---

## 🔬 Methodology

### 1. Data Processing Pipeline

```
Raw Input → Preprocessing → Feature Extraction → Model Inference → Recommendation
```

**Preprocessing Steps:**
- Text normalization and cleaning
- Tokenization and lemmatization
- Stop word removal
- Feature vectorization

### 2. Models & Algorithms

**Primary Model:** Fine-tuned RoBERTa  
**Architecture:** Transformer-based encoder  
**Training:** Supervised learning on emotion-labeled dataset

**Psycholinguistic Features:**
- Sentiment polarity and intensity
- Emotional valence
- Cognitive load indicators
- Language complexity metrics

### 3. Emotion-to-Genre Mapping

Intelligent rule-based system that maps detected emotions to appropriate genres:

| Emotion State | Recommended Genres |
|--------------|-------------------|
| 😊 Happy, Energetic | Adventure, Comedy, Romance |
| 😌 Calm, Peaceful | Poetry, Nature Writing, Spirituality |
| 😔 Sad, Melancholic | Drama, Literary Fiction, Memoir |
| 😰 Anxious, Stressed | Self-Help, Philosophy, Mindfulness |
| 🤔 Curious, Thoughtful | Science, History, Mystery |

---

## 📊 Results

### Performance Metrics

The system was evaluated on a curated dataset with the following results:

<table align="center">
<tr>
<th>Metric</th>
<th>Value</th>
<th>Interpretation</th>
</tr>
<tr>
<td><strong>Accuracy</strong></td>
<td><strong>70.6%</strong></td>
<td>Overall correctness of predictions</td>
</tr>
<tr>
<td><strong>Precision</strong></td>
<td><strong>72.5%</strong></td>
<td>Quality of positive predictions</td>
</tr>
<tr>
<td><strong>Recall</strong></td>
<td><strong>67.0%</strong></td>
<td>Coverage of actual positives</td>
</tr>
<tr>
<td><strong>F1-Score</strong></td>
<td><strong>69.3%</strong></td>
<td>Balanced performance measure</td>
</tr>
</table>

### Key Findings

✅ Emotion-aware recommendations significantly improve relevance compared to behavior-only systems  
✅ The model achieves a strong balance between interpretability and performance  
✅ Psycholinguistic features enhance recommendation accuracy  
✅ System demonstrates feasibility for real-world deployment

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for model downloads)

### Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/DeeptiAttri84/Cognitive-Interaction-Framework.git
cd Cognitive-Interaction-Framework
```

**2. Create virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the application**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 💻 Usage

### Getting Started

1. **Launch the app** using the command above
2. **Choose input method** — Text, Voice, or Image
3. **Express yourself** — Describe your mood, thoughts, or what you're looking for
4. **Receive recommendations** — Get emotionally aligned book suggestions
5. **Explore** — Browse details and find your next read
---

## 📁 Project Structure

```
Cognitive-Interaction-Framework/
│
├── 📄 app.py                    # Streamlit application entry point
├── 🔧 train_model.py            # Model training script
├── 📊 goodreads_data.csv        # Book dataset
├── 🌐 main.html                 # Rendered HTML preview
├── 📋 requirements.txt          # Python dependencies
├── 📜 README.md                 # Documentation
│
├── 🤖 best_model/               # Trained model checkpoints
├── 📝 logs/                     # Training logs and metrics
├── 💾 data/                     # Additional datasets
└── 🖼️ assets/                   # Images and resources
```

---

## 🛠️ Technical Stack

**Core Technologies:**
- Python 3.8+
- PyTorch / TensorFlow
- Transformers (Hugging Face)
- Streamlit

**Key Libraries:**
- `transformers` — Pre-trained language models
- `torch` — Deep learning framework
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `scikit-learn` — Machine learning utilities
- `nltk` — Natural language processing
- `streamlit` — Web interface

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Advanced Model Fine-tuning** — Improve emotion detection accuracy
- [ ] **Multilingual Support** — Detect emotions in multiple languages
- [ ] **Explainable AI** — Provide reasoning behind recommendations
- [ ] **Mobile Application** — iOS and Android apps
- [ ] **User Feedback Loop** — Learn from user preferences over time
- [ ] **Social Features** — Share recommendations with friends
- [ ] **Reading Progress Tracking** — Monitor emotional journey through books
- [ ] **API Integration** — Connect with Goodreads, Amazon, libraries
