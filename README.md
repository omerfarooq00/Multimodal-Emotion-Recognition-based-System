# Multimodal Emotion Recognition Project

This project focuses on building a multimodal emotion recognition system using both text (tweets) and facial expression data. The system combines features from both modalities to predict emotional states more accurately than single-modality approaches.

## Project Structure

```
multimodal-emotion-recognition/
├── data/
│   ├── FER-2013/           # Facial Expression Recognition dataset
│   │   ├── train/          # Training images
│   │   └── test/           # Testing images
│   └── twitter_dataset.csv  # Twitter dataset
├── notebooks/
│   └── emotion_analysis.ipynb  # Jupyter notebook for analysis
├── src/
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── feature_extraction.py  # Feature extraction from text and images
│   ├── model.py              # Model architecture and training
│   └── utils.py              # Utility functions
├── models/                   # Saved models
├── results/                  # Output results and visualizations
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Datasets

### 1. FER-2013 (Facial Expression Recognition)
- Contains 35,887 grayscale images of faces
- 7 emotion categories: angry, disgust, fear, happy, sad, surprise, neutral
- Each image is 48×48 pixels

### 2. Twitter Dataset
- Contains tweets with emotion labels
- Features include:
  - Text content
  - Timestamp
  - Retweets
  - Likes
  - User information

## Features

### Text Features
- Tweet length
- Word count
- Hashtag count
- Mention count
- Sentiment scores
- N-grams

### Image Features
- Facial landmarks
- HOG (Histogram of Oriented Gradients)
- Deep learning features (CNN embeddings)
- Texture features

## Models

### 1. Text Classification
- BERT-based model for text emotion classification
- LSTM/GRU networks
- Traditional ML models (SVM, Random Forest)

### 2. Image Classification
- CNN architectures (VGG, ResNet, EfficientNet)
- Transfer learning with pre-trained models
- Custom CNN architectures

### 3. Multimodal Fusion
- Early fusion: Concatenating features before classification
- Late fusion: Combining model outputs
- Attention mechanisms for feature weighting

## Requirements

- Python 3.8+
- PyTorch / TensorFlow
- Transformers (Hugging Face)
- OpenCV
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data preparation:
```python
from src.data_processing import load_and_preprocess_data

train_loader, val_loader, test_loader = load_and_preprocess_data(
    image_dir="data/FER-2013",
    csv_path="data/twitter_dataset.csv"
)
```

2. Training the model:
```python
from src.model import MultimodalEmotionClassifier

model = MultimodalEmotionClassifier()
model.train(train_loader, val_loader, epochs=50)
```

3. Evaluation:
```python
results = model.evaluate(test_loader)
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

## Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Text-only | 0.65 | 0.64 | 0.63 | 0.64 |
| Image-only | 0.72 | 0.71 | 0.70 | 0.71 |
| Multimodal | 0.83 | 0.82 | 0.83 | 0.82 |

### Key Findings

1. **Multimodal Superiority**
   - The combined model shows significant improvement over single-modality approaches
   - 18% improvement in accuracy compared to image-only model
   - 28% improvement compared to text-only model

2. **Confusion Matrix Analysis**
   - Best performance on 'happy' and 'sad' emotions
   - Some confusion between 'angry' and 'disgust' classes
   - 'Fear' class shows room for improvement

3. **Training Curves**
   - Model converges after ~30 epochs
   - No significant overfitting observed
   - Validation metrics closely follow training metrics

### Result Visualizations

#### 1. Model Performance Comparison

![Model Comparison](results/figures/model_performance/output_10.png)

#### 2. Confusion Matrix

![Confusion Matrix](results/figures/final_results/output_20.png)

#### 3. Training Progress

![Training Curves](results/figures/model_performance/output_15.png)

### Detailed Analysis

For a comprehensive analysis of the results, including additional visualizations and interpretations, please see the [RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md) file in the results directory.

### How to Reproduce

1. Run the Jupyter notebook `FInal_code.ipynb` to generate all results
2. Alternatively, use the provided scripts:
   ```bash
   # Extract images from notebook
   python extract_images.py
   
   # Organize results
   python organize_results.py
   ```
3. View the organized results in the `results/` directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER-2013 dataset
- Hugging Face Transformers
- PyTorch and TensorFlow communities
- All open-source contributors
