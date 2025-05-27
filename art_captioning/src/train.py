import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_data_loaders, setup_metal_device, print_model_info, save_checkpoint
import time


class ArtCaptionTrainer:
    def __init__(self, config):
        print("🚀 ArtCaptionTrainer başlatılıyor...")
        self.config = config
        self.device = setup_metal_device()
        
        # Model ve processor yükle
        self.load_model()
        
        # Data loaders - dataset yapısına optimize edilmiş
        print("\n📊 Data loaders hazırlanıyor...")
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            config['data_dir'], 
            self.processor, 
            config['batch_size']
        )
        
        # Optimizer ve scheduler
        self.setup_optimizer()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        print("✅ ArtCaptionTrainer hazır!")
    
    def load_model(self):
        """Pre-trained BLIP modelini yükle"""
        print("🔄 BLIP modeli yükleniyor...")
        
        # Önce standart BLIP modelini yükle
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Eğer custom checkpoint varsa yükle
        checkpoint_path = self.config['model_path']
        if os.path.exists(checkpoint_path):
            print(f"📂 Custom checkpoint yükleniyor: {checkpoint_path}")
            try:
                # PyTorch checkpoint formatında yükle
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Checkpoint bir dict ise model state_dict al
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        # Dict'in kendisi state_dict olabilir
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    # Checkpoint doğrudan model ise
                    model = checkpoint
                    
                print("✅ Custom checkpoint başarıyla yüklendi!")
            except Exception as e:
                print(f"⚠️ Custom checkpoint yüklenemedi: {e}")
                print("🔄 Standart BLIP modeliyle devam ediliyor...")
        
        self.model = model.to(self.device)
        print_model_info(self.model)
    
    def setup_optimizer(self):
        """Optimizer ve learning rate scheduler ayarla"""
        # Model attributelarını kontrol et
        print("🔍 Model attributeları:")
        for name, _ in self.model.named_children():
            print(f"   - {name}")
        
        # Sadece belirli layer'ları fine-tune et (memory efficiency için)
        trainable_params = []
        
        # Vision encoder'ı dondurmaya bırak (memory tasarrufu)
        if hasattr(self.model, 'vision_model'):
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
        
        # Text decoder varsa fine-tune et
        if hasattr(self.model, 'text_decoder'):
            for param in self.model.text_decoder.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        
        # Language model varsa fine-tune et
        if hasattr(self.model, 'language_model'):
            for param in self.model.language_model.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        
        # Eğer hiç trainable param yoksa tüm modeli train et
        if len(trainable_params) == 0:
            print("⚠️ Hiç trainable parametre bulunamadı, tüm modeli fine-tune ediliyor...")
            trainable_params = list(self.model.parameters())
            for param in trainable_params:
                param.requires_grad = True
        
        print(f"🎯 Fine-tune edilecek parametre sayısı: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = AdamW(trainable_params, lr=self.config['learning_rate'], weight_decay=0.01)
        
        # Cosine scheduler
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(0.1 * total_steps)  # İlk %10'da warm-up
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, epoch):
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Batch'i device'a taşı
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Loss None kontrolü
                if loss is None:
                    print(f"⚠️ Batch {batch_idx} - Loss is None, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # Progress bar güncelle
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Metal GPU memory management
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} hatası: {e}")
                continue
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                except Exception as e:
                    print(f"⚠️ Validation batch hatası: {e}")
                    continue
        
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
    
    def generate_sample_captions(self, num_samples=3):
        """Örnek caption'lar generate et"""
        self.model.eval()
        print("\n🎨 Örnek Caption'lar:")
        print("-" * 50)
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                
                try:
                    # İlk görseli al
                    pixel_values = batch['pixel_values'][:1].to(self.device)
                    
                    # Caption generate et
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=50,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    generated_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # Original caption
                    if 'labels' in batch:
                        original_ids = batch['labels'][:1]
                        original_caption = self.processor.decode(original_ids[0], skip_special_tokens=True)
                        print(f"Örnek {i+1}:")
                        print(f"  Orijinal: {original_caption}")
                        print(f"  Generated: {generated_caption}")
                    else:
                        print(f"Örnek {i+1} Generated: {generated_caption}")
                    
                    print()
                
                except Exception as e:
                    print(f"⚠️ Caption generation hatası: {e}")
                    continue
    
    def train(self):
        """Ana training loop"""
        print("🚀 Training başlıyor...")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"📊 Batch size: {self.config['batch_size']}")
        print(f"📊 Epochs: {self.config['epochs']}")
        print(f"📊 Learning rate: {self.config['learning_rate']}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\n🔄 Epoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"📈 Train Loss: {train_loss:.4f}")
            print(f"📉 Val Loss: {val_loss:.4f}")
            
            # En iyi modeli kaydet
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, best_model_path)
            
            # Her epoch sonunda örnek caption'lar göster
            if (epoch + 1) % 1 == 0:
                self.generate_sample_captions()
            
            # Checkpoint kaydet
            checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
        
        # Training tamamlandı
        print("\n🎉 Training tamamlandı!")
        print(f"💾 En iyi model: {best_model_path}")
        
        # Loss grafiği çiz
        self.plot_training_history()
    
    def plot_training_history(self):
        """Training history grafiği"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Art Caption Training History')
        plt.legend()
        plt.grid(True)
        
        # Grafik kaydet
        plot_path = os.path.join(self.config['output_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training grafiği kaydedildi: {plot_path}")
        plt.close()


def main():
    """Ana fonksiyon"""
    # Config
    config = {
        'data_dir': 'data',
        'model_path': 'models/model_base_capfilt_large.pth',
        'output_dir': 'outputs',
        'batch_size': 4,  # Dataset optimize edildi, batch size artırıldı
        'learning_rate': 1e-4,  # Learning rate artırıldı (0.0001) - art data için optimum
        'epochs': 5,
    }
    
    # File path'leri detaylı print et
    print("\n📁 KULLANILAN DOSYA YOLLARI:")
    print(f"  🗂️  Data klasörü: {os.path.abspath(config['data_dir'])}")
    print(f"  📊 JSON dosyası: {os.path.abspath(os.path.join(config['data_dir'], 'ArtCap.json'))}")
    print(f"  🖼️  Görsel klasörü: {os.path.abspath(os.path.join(config['data_dir'], 'ArtCap_Images_Dataset'))}")
    print(f"  🤖 Model dosyası: {os.path.abspath(config['model_path'])}")
    print(f"  💾 Output klasörü: {os.path.abspath(config['output_dir'])}")
    
    # File kontrolü
    json_path = os.path.join(config['data_dir'], 'ArtCap.json')
    images_dir = os.path.join(config['data_dir'], 'ArtCap_Images_Dataset')
    model_path = config['model_path']
    
    print(f"\n🔍 DOSYA KONTROLLERI:")
    print(f"  {'✅' if os.path.exists(json_path) else '❌'} JSON dosyası mevcut: {json_path}")
    print(f"  {'✅' if os.path.exists(images_dir) else '❌'} Görsel klasörü mevcut: {images_dir}")
    print(f"  {'✅' if os.path.exists(model_path) else '❌'} Model dosyası mevcut: {model_path}")
    
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  📊 Görsel sayısı: {image_count}")
    
    print(f"\n⚙️ EĞİTİM PARAMETRELERİ:")
    print(f"  📦 Batch Size: {config['batch_size']}")
    print(f"  📈 Learning Rate: {config['learning_rate']}")
    print(f"  🔄 Epochs: {config['epochs']}")
    print(f"  💾 Output Klasörü: {config['output_dir']}")
    print("=" * 60)
    
    # Output dizini oluştur
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Trainer oluştur ve training başlat
    trainer = ArtCaptionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 