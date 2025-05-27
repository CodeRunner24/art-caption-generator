import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor
import numpy as np
from typing import Dict, List, Tuple


class ArtCapDataset(Dataset):
    """Art caption dataset for BLIP fine-tuning - Optimized for ArtCap format"""
    
    def __init__(self, json_path: str, images_dir: str, processor, split_ratio=None, split_type='train'):
        """
        Args:
            json_path: ArtCap.json dosyasının yolu  
            images_dir: Görsellerin bulunduğu klasör
            processor: BLIP processor
            split_ratio: (train, val, test) oranları, default (0.8, 0.1, 0.1)
            split_type: 'train', 'val', veya 'test'
        """
        self.images_dir = images_dir
        self.processor = processor
        
        if split_ratio is None:
            split_ratio = (0.8, 0.1, 0.1)
        
        # JSON dosyasını yükle
        print(f"📂 {json_path} yükleniyor...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Dataset verification
        print(f"✅ JSON'da {len(self.data)} görsel bulundu")
        
        # Mevcut görselleri kontrol et
        available_images = set(os.listdir(images_dir))
        print(f"✅ Klasörde {len(available_images)} görsel bulundu")
        
        # Her görsel için tüm caption'ları düzenle
        self.image_caption_pairs = []
        missing_images = []
        
        for image_name, captions in self.data.items():
            if image_name in available_images:
                # Her caption için ayrı entry oluştur
                for caption in captions:
                    if caption.strip():  # Boş caption'ları atla
                        self.image_caption_pairs.append((image_name, caption.strip()))
            else:
                missing_images.append(image_name)
        
        if missing_images:
            print(f"⚠️ {len(missing_images)} görsel klasörde bulunamadı")
        
        print(f"✅ Toplam {len(self.image_caption_pairs)} image-caption çifti hazırlandı")
        
        # Reproducible split için seed ayarla
        random.seed(42)
        random.shuffle(self.image_caption_pairs)
        
        # Data split
        total_len = len(self.image_caption_pairs)
        train_end = int(total_len * split_ratio[0])
        val_end = train_end + int(total_len * split_ratio[1])
        
        if split_type == 'train':
            self.image_caption_pairs = self.image_caption_pairs[:train_end]
        elif split_type == 'val':
            self.image_caption_pairs = self.image_caption_pairs[train_end:val_end]
        elif split_type == 'test':
            self.image_caption_pairs = self.image_caption_pairs[val_end:]
        
        print(f"🎯 {split_type.upper()} dataset: {len(self.image_caption_pairs)} örnekle")
        
        # Cache görsellerin boyutlarını kontrol et (ilk 5 örnek)
        self._check_sample_images()
    
    def _check_sample_images(self):
        """İlk birkaç görseli kontrol et"""
        print(f"🔍 Örnek görselleri kontrol ediliyor...")
        
        for i in range(min(3, len(self.image_caption_pairs))):
            image_name, caption = self.image_caption_pairs[i]
            try:
                image_path = os.path.join(self.images_dir, image_name)
                image = Image.open(image_path)
                print(f"  ✅ {image_name}: {image.size} - {caption[:50]}...")
            except Exception as e:
                print(f"  ❌ {image_name}: HATA - {e}")
    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_name, caption = self.image_caption_pairs[idx]
        
        # Görseli yükle
        try:
            image_path = os.path.join(self.images_dir, image_name)
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Görsel yükleme hatası {image_name}: {e}")
            # Boş beyaz görsel döndür
            image = Image.new('RGB', (384, 384), color='white')
        
        return {
            'image': image,
            'caption': caption,
            'image_name': image_name
        }


def collate_fn(batch, processor):
    """BLIP fine-tuning için optimize edilmiş collate function"""
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    try:
        # Image preprocessing
        image_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Text preprocessing - BLIP için özel format
        text_inputs = processor.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64  # Art caption'lar genelde kısa (avg 11 kelime)
        )
        
        # BLIP training batch format
        return {
            'pixel_values': image_inputs['pixel_values'],
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': text_inputs['input_ids'].clone()  # Loss hesabı için
        }
        
    except Exception as e:
        print(f"⚠️ Collate function hatası: {e}")
        # Fallback: boş batch döndür
        batch_size = len(batch)
        return {
            'pixel_values': torch.zeros(batch_size, 3, 384, 384),
            'input_ids': torch.zeros(batch_size, 10, dtype=torch.long),
            'attention_mask': torch.zeros(batch_size, 10, dtype=torch.long),
            'labels': torch.zeros(batch_size, 10, dtype=torch.long)
        }


def get_data_loaders(data_dir: str, processor, batch_size: int = 4):
    """Optimize edilmiş data loader'lar"""
    
    json_path = os.path.join(data_dir, 'ArtCap.json')
    images_dir = os.path.join(data_dir, 'ArtCap_Images_Dataset')
    
    print(f"📊 Data loaders hazırlanıyor...")
    print(f"  📁 JSON: {json_path}")
    print(f"  📁 Görseller: {images_dir}")
    print(f"  📦 Batch size: {batch_size}")
    
    # Dataset'leri oluştur
    train_dataset = ArtCapDataset(json_path, images_dir, processor, split_type='train')
    val_dataset = ArtCapDataset(json_path, images_dir, processor, split_type='val')
    test_dataset = ArtCapDataset(json_path, images_dir, processor, split_type='test')
    
    # Custom collate function
    def collate_wrapper(batch):
        return collate_fn(batch, processor)
    
    # DataLoader'ları oluştur - Mac optimizasyonu
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Mac'te daha stabil
        pin_memory=False,  # MPS ile pin_memory problem çıkarabiliyor
        collate_fn=collate_wrapper,
        drop_last=True  # Son incomplete batch'i atla
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=False,
        collate_fn=collate_wrapper,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=False,
        collate_fn=collate_wrapper,
        drop_last=False
    )
    
    print(f"✅ Data loaders hazır:")
    print(f"  🚂 Train: {len(train_loader)} batch ({len(train_dataset)} örnek)")
    print(f"  ✅ Val: {len(val_loader)} batch ({len(val_dataset)} örnek)")
    print(f"  🧪 Test: {len(test_loader)} batch ({len(test_dataset)} örnek)")
    
    return train_loader, val_loader, test_loader


def setup_metal_device():
    """Metal GPU ayarlarını optimize et"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Metal Performance Shaders (MPS) kullanılıyor!")
        print(f"   MPS Backend: {torch.backends.mps.is_built()}")
        
        # MPS için optimizasyonlar
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ NVIDIA CUDA GPU kullanılıyor!")
        print(f"   GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU kullanılıyor - GPU'ya göre yavaş olacak!")
    
    return device


def print_model_info(model):
    """Model bilgilerini detaylı yazdır"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n🤖 Model Architecture Bilgileri:")
    print(f"  📊 Toplam parametre: {total_params:,}")
    print(f"  🎯 Eğitilecek parametre: {trainable_params:,}")
    print(f"  🧊 Dondurulmuş parametre: {frozen_params:,}")
    print(f"  📈 Eğitilecek oran: {100 * trainable_params / total_params:.1f}%")
    
    # Model komponenleri
    print(f"\n🏗️ Model Komponenleri:")
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        status = "🎯 TRAINABLE" if child_trainable > 0 else "🧊 FROZEN"
        print(f"  {status} {name}: {child_params:,} params")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Checkpoint kaydetme (metadata ile)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    import time
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_type': 'BLIP_ArtCaption_FineTuned',
        'timestamp': time.time()  # Unix timestamp
    }
    
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint kaydedildi: {path}")
    print(f"   📊 Epoch: {epoch}, Loss: {loss:.4f}")


def load_checkpoint(model, optimizer, path):
    """Checkpoint yükleme (hata kontrolü ile)"""
    try:
        print(f"📂 Checkpoint yükleniyor: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"✅ Checkpoint yüklendi!")
        print(f"   📊 Epoch: {epoch}, Loss: {loss:.4f}")
        
        return epoch, loss
        
    except Exception as e:
        print(f"⚠️ Checkpoint yükleme hatası: {e}")
        return 0, float('inf') 