# 🎨 Art Caption Generator

BLIP modelini sanat eserleri üzerinde fine-tune ederek art-specific image captioning sistemi.

## 🚀 Özellikler

- **Fine-tuned BLIP Model**: Sanat eserleri için özelleştirilmiş
- **Metal GPU Support**: Apple Silicon optimizasyonu
- **Interactive Demo**: Gradio web interface
- **Multiple Generation Modes**: Conservative, Balanced, Creative
- **Art Terminology**: Sanat-specific kelime kullanımı
- **Comprehensive Testing**: Model performance analizi

## 📁 Proje Yapısı

```
art_captioning/
├── data/
│   ├── ArtCap.json                 # Dataset annotations
│   └── ArtCap_Images_Dataset/      # Art images
├── models/
│   └── model_base_capfilt_large.pth  # Pre-trained BLIP model
├── src/
│   ├── utils.py                    # Dataset & utilities
│   ├── train.py                    # Training script
│   ├── test.py                     # Testing script
│   └── demo.py                     # Gradio demo
├── outputs/                        # Training outputs
├── main.py                         # Main entry point
├── requirements.txt                # Dependencies
└── README.md                       # Bu dosya
```

## 🛠️ Kurulum

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Çevre Kurulumu

```bash
python art_captioning/main.py setup
```

### 3. Proje Durumunu Kontrol Et

```bash
python art_captioning/main.py status
```

## 🎯 Kullanım

### Model Eğitimi

```bash
# Fine-tuning başlat
python art_captioning/main.py train

# Eğitim parametreleri (src/train.py içinde):
# - Batch size: 3 (Metal GPU için optimize)
# - Learning rate: 5e-5
# - Epochs: 5
# - Optimizer: AdamW
```

### Model Test

```bash
# Trained modeli test et
python art_captioning/main.py test

# Test özellikleri:
# - Rastgele örnekler
# - Art terminology analizi
# - Baseline karşılaştırma
# - Görsel sonuçlar
```

### Interactive Demo

```bash
# Gradio demo başlat
python art_captioning/main.py demo

# Demo özellikleri:
# - Web interface (localhost:7860)
# - Çoklu generation modu
# - Gerçek zamanlı caption
# - Art analysis
```

## 🎨 Generation Modları

### Conservative Mode
- **Algoritma**: Beam search
- **Özellik**: Güvenilir ve doğru sonuçlar
- **Kullanım**: Akademik veya profesyonel analiz

### Balanced Mode
- **Algoritma**: Beam + Sampling hybrid
- **Özellik**: Kalite ve çeşitlilik dengesi
- **Kullanım**: Genel amaçlı caption

### Creative Mode
- **Algoritma**: High-temperature sampling
- **Özellik**: Yaratıcı ve özgün açıklamalar
- **Kullanım**: Sanatsal yorumlama

## 📊 Model Performansı

### Art Terminology Kullanımı
- **Style terms**: abstract, realistic, impressionist
- **Medium terms**: oil painting, watercolor, acrylic
- **Composition**: foreground, background, perspective
- **Color**: palette, vibrant, contrast, hue
- **Technique**: brushstroke, texture, chiaroscuro

### Hardware Optimizasyonu
- **Metal GPU**: Apple Silicon M1/M2/M3 support
- **Memory Efficient**: Selective layer fine-tuning
- **Batch Processing**: Optimal batch sizes for local GPU

## 🔧 Teknik Detaylar

### Model Architecture
- **Base Model**: Salesforce/blip-image-captioning-base
- **Fine-tuning Strategy**: Decoder + last encoder layers
- **Frozen Layers**: Vision encoder (memory efficiency)
- **Total Parameters**: ~400M (fine-tuned: ~120M)

### Training Configuration
```python
config = {
    'batch_size': 3,           # Metal GPU optimized
    'learning_rate': 5e-5,     # Conservative learning
    'epochs': 5,               # Prevent overfitting
    'warmup_steps': 10%,       # Gradual learning
    'scheduler': 'cosine',     # Smooth decay
    'gradient_clipping': 1.0   # Stability
}
```

### Dataset Statistics
- **Total Images**: ~100+ art pieces
- **Total Captions**: ~500+ descriptions
- **Split Ratio**: 80% train, 10% val, 10% test
- **Caption Length**: Average 50-100 words

## 🚨 Troubleshooting

### Memory Issues
```bash
# Metal GPU memory limit exceeded
# Solution: Reduce batch size in train.py
batch_size = 2  # or even 1
```

### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# Path issues
export PYTHONPATH="${PYTHONPATH}:./art_captioning/src"
```

### Model Loading Errors
```bash
# Checkpoint format issues
# Check: art_captioning/outputs/best_model.pth exists
# Fallback: Uses base BLIP model automatically
```

## 🎯 Sonuçlar ve Değerlendirme

### Expected Improvements
- **Art-specific vocabulary** kullanımında artış
- **Style recognition** performansında iyileşme
- **Composition analysis** detayında artış
- **Color description** zenginliğinde gelişme

### Evaluation Metrics
- **BLEU Score**: Caption quality measurement
- **Art Terminology Rate**: Domain-specific word usage
- **Human Evaluation**: Subjective quality assessment
- **Inference Speed**: Generation time optimization

## 📝 Lisans ve Kullanım

Bu proje eğitim amaçlı olarak geliştirilmiştir. Ticari kullanım için:
- BLIP model lisansını kontrol edin
- Dataset telif haklarını göz önünde bulundurun
- Attribution gereklilikleri yerine getirin

## 🤝 Katkıda Bulunma

1. **Fork** this repository
2. **Create** feature branch
3. **Make** improvements
4. **Test** thoroughly
5. **Submit** pull request

## 📞 İletişim ve Destek

- **Issues**: GitHub issues üzerinden
- **Improvements**: Feature requests welcome
- **Documentation**: README updates appreciated

---

### 🎨 Sample Outputs

**Original Caption**: "a man is sitting in a chair"
**Fine-tuned Caption**: "an impressionist oil painting depicting a gentleman in formal attire seated on an ornate wooden chair, rendered with visible brushstrokes and warm palette"

**Conservation Analysis**: "Renaissance-style portrait painting featuring classical composition"
**Creative Analysis**: "A masterful study in light and shadow, capturing the essence of human dignity through masterful brushwork" 