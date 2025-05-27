# 🎨 Art Caption Generator

A fine-tuned BLIP model for generating descriptive captions of artwork images, optimized for art-specific terminology and visual analysis.

## ✨ Features

- 🎨 **Fine-tuned BLIP Model**: Specialized for art image captioning
- 🖼️ **Multiple Generation Modes**: Conservative, Balanced, and Creative
- 🔍 **Advanced Analysis**: Multi-perspective artwork analysis
- 📋 **Clipboard Support**: Paste images directly (Ctrl+V / Cmd+V)
- 🌐 **Interactive Demo**: User-friendly Gradio interface
- ⚡ **Metal GPU Support**: Optimized for Apple Silicon (M1/M2/M3)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/art-caption-generator.git
cd art-caption-generator

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Create the following structure:
```
art_captioning/
├── data/
│   ├── ArtCap.json                 # Captions in JSON format
│   └── ArtCap_Images_Dataset/      # Image files
└── models/
    └── model_base_capfilt_large.pth  # Pre-trained BLIP model
```

**Dataset Format (ArtCap.json):**
```json
{
  "image1.jpg": [
    "caption 1 for image1",
    "caption 2 for image1",
    "caption 3 for image1"
  ],
  "image2.jpg": [
    "caption 1 for image2",
    "caption 2 for image2"
  ]
}
```

### 3. Training

```bash
cd art_captioning
python src/train.py
```

### 4. Testing

```bash
python src/test.py
```

### 5. Demo

```bash
python src/demo.py
```

## 📊 Model Performance

Our fine-tuned model achieves:
- **Training Loss**: 0.45 (final epoch)
- **Validation Loss**: 1.98 (best model at epoch 2)
- **Generation Speed**: ~2.3 it/s on Apple M3

### Training Progress
- **Epoch 1**: Train: 2.31 → Val: 2.06
- **Epoch 2**: Train: 1.75 → Val: 1.98 ✅ (Best Model)
- **Epoch 3**: Train: 1.26 → Val: 2.02
- **Epoch 4**: Train: 0.76 → Val: 2.16
- **Epoch 5**: Train: 0.45 → Val: 2.30

## 🎯 Generation Modes

### Conservative Mode
- Uses beam search for reliable results
- More factual and straightforward descriptions
- Best for accuracy-focused applications

### Balanced Mode  
- Hybrid approach (beam + sampling)
- Good balance between creativity and accuracy
- Recommended for most use cases

### Creative Mode
- High temperature sampling
- More unique and artistic descriptions
- Best for creative and experimental captions

## 📁 Project Structure

```
art_captioning/
├── src/
│   ├── train.py          # Training script
│   ├── test.py           # Model evaluation
│   ├── demo.py           # Gradio interface
│   └── utils.py          # Helper functions
├── data/                 # Dataset (not included)
├── models/               # Model files (not included)
├── outputs/              # Training outputs
├── requirements.txt      # Dependencies
└── README.md
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: Salesforce/blip-image-captioning-base
- **Parameters**: 247M total (161M trainable)
- **Fine-tuning**: Text decoder only (vision encoder frozen)
- **Optimization**: AdamW with cosine scheduler

### Training Configuration
- **Dataset**: 18,030 image-caption pairs
- **Split**: 80% train, 10% validation, 10% test  
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Epochs**: 5
- **Hardware**: Apple M3 with Metal Performance Shaders

### Supported Features
- Art terminology recognition
- Style identification (abstract, realistic, impressionist)
- Technique description (oil painting, watercolor, brushstroke)
- Composition analysis (foreground, background, perspective)

## 🔧 Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
gradio>=3.35.0
Pillow>=9.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
numpy>=1.24.0
```

## 💻 Hardware Requirements

### Minimum
- RAM: 8GB
- Storage: 5GB free space
- GPU: Optional (CPU supported)

### Recommended  
- RAM: 16GB+
- GPU: Apple Silicon (M1/M2/M3) or NVIDIA GPU
- Storage: 10GB free space

## 🚀 Usage Examples

### Basic Caption Generation
```python
from src.demo import ArtCaptionDemo
from PIL import Image

# Load model
demo = ArtCaptionDemo("outputs/best_model.pth")

# Generate caption
image = Image.open("artwork.jpg")
caption = demo.generate_caption(image, generation_mode="Balanced")
print(caption)
```

### Advanced Analysis
```python
# Multi-perspective analysis
analysis = demo.analyze_art_features(image)
print(analysis)
```

## 📈 Results Examples

**Input**: Van Gogh's Starry Night  
**Output**: "a painting of a swirling night sky with bright stars over a village with a prominent church spire"

**Input**: Picasso's Les Demoiselles d'Avignon  
**Output**: "an abstract painting featuring geometric faces and fragmented forms in earth tones"

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Salesforce BLIP**: Base model architecture
- **Hugging Face Transformers**: Model implementation
- **Gradio**: Interactive demo interface
- **Apple Metal**: GPU acceleration on Apple Silicon

## 📞 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com]

---

⭐ If you find this project helpful, please give it a star on GitHub! 