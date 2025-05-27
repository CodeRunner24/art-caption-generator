import os
import torch
from PIL import Image
import random
from transformers import BlipForConditionalGeneration, BlipProcessor
from utils import setup_metal_device, ArtCapDataset
import matplotlib.pyplot as plt
import numpy as np


class ArtCaptionTester:
    def __init__(self, model_path=None, data_dir='data'):
        self.device = setup_metal_device()
        self.data_dir = data_dir
        
        # Model ve processor yükle
        self.load_model(model_path)
        
        # Test dataset
        self.load_test_data()
    
    def load_model(self, model_path=None):
        """Model ve processor yükle"""
        print("🔄 Model yükleniyor...")
        
        # Processor yükle
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        if model_path and os.path.exists(model_path):
            print(f"📂 Fine-tuned model yükleniyor: {model_path}")
            try:
                # Checkpoint yükle
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Standart BLIP modelini yükle
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                
                # Fine-tuned weights yükle
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("✅ Fine-tuned model başarıyla yüklendi!")
                else:
                    print("⚠️ Checkpoint formatı tanınmadı, standart BLIP kullanılıyor")
                    
            except Exception as e:
                print(f"⚠️ Fine-tuned model yüklenemedi: {e}")
                print("🔄 Standart BLIP modeliyle devam ediliyor...")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        else:
            print("🔄 Standart BLIP modeli yükleniyor...")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def load_test_data(self):
        """Test dataset yükle"""
        json_path = os.path.join(self.data_dir, 'ArtCap.json')
        images_dir = os.path.join(self.data_dir, 'ArtCap_Images_Dataset')
        
        self.test_dataset = ArtCapDataset(
            json_path, images_dir, self.processor, split_type='test'
        )
        print(f"📊 Test dataset: {len(self.test_dataset)} örnek")
    
    def generate_caption(self, image, use_beam_search=True, temperature=0.7):
        """Tek bir görsel için caption generate et"""
        with torch.no_grad():
            # Görsel preprocessing
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            if use_beam_search:
                # Beam search ile daha kaliteli sonuçlar
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0
                )
            else:
                # Sampling ile daha çeşitli sonuçlar
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=100,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    early_stopping=True
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
    
    def test_random_samples(self, num_samples=10):
        """Rastgele örnekler üzerinde test"""
        print(f"\n🎨 {num_samples} Rastgele Örnek Test Ediliyor...")
        print("=" * 80)
        
        # Rastgele indeksler seç
        indices = random.sample(range(len(self.test_dataset)), min(num_samples, len(self.test_dataset)))
        
        results = []
        
        for i, idx in enumerate(indices):
            print(f"\n📷 Örnek {i+1}/{num_samples}")
            print("-" * 40)
            
            try:
                # Dataset'ten örnek al
                sample = self.test_dataset[idx]
                image_name = self.test_dataset.image_caption_pairs[idx][0]
                original_caption = self.test_dataset.image_caption_pairs[idx][1]
                
                # Görseli yükle
                image_path = os.path.join(self.data_dir, 'ArtCap_Images_Dataset', image_name)
                image = Image.open(image_path).convert('RGB')
                
                # Caption generate et
                print("🔄 Caption generate ediliyor...")
                generated_caption = self.generate_caption(image)
                
                print(f"🖼️ Görsel: {image_name}")
                print(f"📝 Orijinal: {original_caption}")
                print(f"🤖 Generated: {generated_caption}")
                
                # Sonuçları kaydet
                results.append({
                    'image_name': image_name,
                    'original': original_caption,
                    'generated': generated_caption,
                    'image': image
                })
                
            except Exception as e:
                print(f"⚠️ Hata: {e}")
                continue
        
        return results
    
    def test_specific_image(self, image_path):
        """Belirli bir görsel üzerinde test"""
        print(f"\n🖼️ Test ediliyor: {image_path}")
        print("-" * 50)
        
        try:
            # Görseli yükle
            image = Image.open(image_path).convert('RGB')
            
            # Farklı generation stratejileri test et
            print("🔄 Caption'lar generate ediliyor...")
            
            beam_caption = self.generate_caption(image, use_beam_search=True)
            sample_caption = self.generate_caption(image, use_beam_search=False, temperature=0.7)
            creative_caption = self.generate_caption(image, use_beam_search=False, temperature=1.0)
            
            print(f"🎯 Beam Search: {beam_caption}")
            print(f"🎲 Sampling (T=0.7): {sample_caption}")
            print(f"🎨 Creative (T=1.0): {creative_caption}")
            
            return {
                'beam': beam_caption,
                'sample': sample_caption,
                'creative': creative_caption,
                'image': image
            }
            
        except Exception as e:
            print(f"⚠️ Hata: {e}")
            return None
    
    def evaluate_art_terminology(self, results):
        """Art-specific terminoloji kullanımını değerlendir"""
        print("\n🎨 Art Terminoloji Analizi")
        print("=" * 50)
        
        # Art-specific terimler
        art_terms = {
            'style': ['abstract', 'realistic', 'impressionist', 'expressionist', 'cubist', 'surreal'],
            'medium': ['painting', 'oil', 'watercolor', 'acrylic', 'canvas', 'brush'],
            'composition': ['foreground', 'background', 'composition', 'perspective', 'depth'],
            'color': ['palette', 'hue', 'saturation', 'contrast', 'vibrant', 'muted'],
            'subject': ['portrait', 'landscape', 'still life', 'figure', 'nude'],
            'technique': ['brushstroke', 'texture', 'shading', 'lighting', 'chiaroscuro']
        }
        
        # Term kullanım istatistikleri
        term_usage = {}
        total_captions = len(results)
        
        for category, terms in art_terms.items():
            category_count = 0
            for result in results:
                caption = result['generated'].lower()
                if any(term in caption for term in terms):
                    category_count += 1
            
            usage_rate = (category_count / total_captions) * 100
            term_usage[category] = usage_rate
            print(f"📊 {category.title()}: {usage_rate:.1f}% ({category_count}/{total_captions})")
        
        # Ortalama art terminology kullanımı
        avg_usage = np.mean(list(term_usage.values()))
        print(f"\n🎯 Ortalama Art Terminology Kullanımı: {avg_usage:.1f}%")
        
        return term_usage
    
    def create_results_visualization(self, results, save_path='outputs/test_results.png'):
        """Test sonuçlarını görselleştir"""
        if not results:
            print("⚠️ Görselleştirilecek sonuç yok")
            return
        
        num_results = min(6, len(results))  # Maksimum 6 örnek göster
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎨 Art Caption Generation Results', fontsize=16, fontweight='bold')
        
        for i in range(num_results):
            row = i // 3
            col = i % 3
            
            if i < len(results):
                result = results[i]
                
                # Görseli göster
                axes[row, col].imshow(result['image'])
                axes[row, col].axis('off')
                
                # Caption'ları title olarak ekle
                title = f"🤖 {result['generated'][:50]}...\n📝 {result['original'][:50]}..."
                axes[row, col].set_title(title, fontsize=10, pad=10, wrap=True)
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Test sonuçları kaydedildi: {save_path}")
        plt.close()
    
    def compare_with_baseline(self, num_samples=20):
        """Fine-tuned model ile baseline modeli karşılaştır"""
        print("\n⚖️ Baseline Karşılaştırma")
        print("=" * 50)
        
        # Baseline model yükle
        print("🔄 Baseline BLIP model yükleniyor...")
        baseline_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        baseline_model = baseline_model.to(self.device)
        baseline_model.eval()
        
        # Rastgele örnekler seç
        indices = random.sample(range(len(self.test_dataset)), min(num_samples, len(self.test_dataset)))
        
        comparisons = []
        
        for i, idx in enumerate(indices):
            try:
                sample = self.test_dataset[idx]
                image_name = self.test_dataset.image_caption_pairs[idx][0]
                
                # Görseli yükle
                image_path = os.path.join(self.data_dir, 'ArtCap_Images_Dataset', image_name)
                image = Image.open(image_path).convert('RGB')
                
                # Fine-tuned model caption
                fine_tuned_caption = self.generate_caption(image)
                
                # Baseline model caption
                with torch.no_grad():
                    inputs = self.processor(images=image, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].to(self.device)
                    
                    generated_ids = baseline_model.generate(
                        pixel_values,
                        max_length=100,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    baseline_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                comparisons.append({
                    'image_name': image_name,
                    'fine_tuned': fine_tuned_caption,
                    'baseline': baseline_caption
                })
                
                if i < 5:  # İlk 5 örneği göster
                    print(f"\n📷 {image_name}")
                    print(f"🎨 Fine-tuned: {fine_tuned_caption}")
                    print(f"🔘 Baseline: {baseline_caption}")
                
            except Exception as e:
                print(f"⚠️ Karşılaştırma hatası: {e}")
                continue
        
        return comparisons


def main():
    """Ana test fonksiyonu"""
    print("🧪 Art Caption Model Test")
    print("=" * 50)
    
    # Model yolunu belirle
    model_path = "outputs/best_model.pth"
    if not os.path.exists(model_path):
        print("⚠️ Fine-tuned model bulunamadı, standart BLIP kullanılacak")
        model_path = None
    
    # Tester oluştur
    tester = ArtCaptionTester(model_path)
    
    # 1. Rastgele örnekler test et
    results = tester.test_random_samples(10)
    
    # 2. Art terminology analizi
    if results:
        tester.evaluate_art_terminology(results)
    
    # 3. Sonuçları görselleştir
    if results:
        tester.create_results_visualization(results)
    
    # 4. Baseline ile karşılaştır (eğer fine-tuned model varsa)
    if model_path:
        tester.compare_with_baseline(10)
    
    print("\n✅ Test tamamlandı!")


if __name__ == "__main__":
    main() 