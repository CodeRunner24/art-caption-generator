import gradio as gr
import torch
import os
from PIL import Image
import numpy as np
from transformers import BlipForConditionalGeneration, BlipProcessor
from utils import setup_metal_device
import time


class ArtCaptionDemo:
    def __init__(self, model_path=None):
        """Demo için model yükle"""
        print("🎨 Art Caption Demo başlatılıyor...")
        
        self.device = setup_metal_device()
        self.load_model(model_path)
        
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
                    print("✅ Fine-tuned model yüklendi!")
                    self.is_fine_tuned = True
                else:
                    print("⚠️ Checkpoint formatı tanınmadı")
                    self.is_fine_tuned = False
                    
            except Exception as e:
                print(f"⚠️ Fine-tuned model yüklenemedi: {e}")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.is_fine_tuned = False
        else:
            print("🔄 Standart BLIP modeli yükleniyor...")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.is_fine_tuned = False
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model hazır!")
    
    def generate_caption(self, image, generation_mode="Balanced", creativity=0.7):
        """Generate caption for image"""
        if image is None:
            return "⚠️ Please upload an image!"
        
        try:
            # Görsel preprocessing
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # PIL Image'a çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            start_time = time.time()
            
            with torch.no_grad():
                # Processor ile preprocess
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                # Generation parametreleri
                if generation_mode == "Conservative":
                    # Beam search ile güvenilir sonuçlar
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=100,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0
                    )
                elif generation_mode == "Creative":
                    # Sampling ile yaratıcı sonuçlar
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=120,
                        do_sample=True,
                        temperature=creativity,
                        top_p=0.9,
                        top_k=50,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                else:  # Balanced
                    # Karma yaklaşım
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=100,
                        num_beams=3,
                        do_sample=True,
                        temperature=creativity,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                
                # Caption decode et
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                generation_time = time.time() - start_time
                
                # Sonuç formatla
                model_type = "🎨 Fine-tuned Art Model" if self.is_fine_tuned else "🔘 Base BLIP Model"
                result = f"**{model_type}**\n\n📝 **Caption:** {caption}\n\n⏱️ **Generation Time:** {generation_time:.2f}s"
                
                return result
                
        except Exception as e:
            return f"⚠️ Error occurred: {str(e)}"
    
    def analyze_art_features(self, image):
        """Analyze art features"""
        if image is None:
            return "⚠️ Please upload an image!"
        
        try:
            # Farklı generation stratejileri ile analiz
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                # Konservatif analiz
                conservative_ids = self.model.generate(
                    pixel_values,
                    max_length=80,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                conservative_caption = self.processor.decode(conservative_ids[0], skip_special_tokens=True)
                
                # Yaratıcı analiz
                creative_ids = self.model.generate(
                    pixel_values,
                    max_length=100,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.9,
                    early_stopping=True
                )
                creative_caption = self.processor.decode(creative_ids[0], skip_special_tokens=True)
                
                # Detaylı analiz
                detailed_ids = self.model.generate(
                    pixel_values,
                    max_length=120,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.8,
                    early_stopping=True
                )
                detailed_caption = self.processor.decode(detailed_ids[0], skip_special_tokens=True)
                
                # Sonuçları formatla
                analysis = f"""
## 🎨 Art Analysis Results

### 🎯 **Conservative Analysis**
*Trusted and accurate description*
{conservative_caption}

### 🌟 **Creative Analysis** 
*Creative and unique approach*
{creative_caption}

### 📖 **Detailed Analysis**
*Comprehensive and detailed description*
{detailed_caption}

### 📊 **Model Info**
**Type:** {"Fine-tuned Art Model" if self.is_fine_tuned else "Base BLIP Model"}
**Device:** {self.device.type.upper()}
"""
                return analysis
                
        except Exception as e:
                            return f"⚠️ Analysis error: {str(e)}"
    
    def create_interface(self):
        """Gradio interface oluştur"""
        
        # CSS stili
        css = """
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .art-container {
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            padding: 15px;
            background: #f8f9fa;
        }
        """
        
        with gr.Blocks(css=css, title="🎨 Art Caption Generator") as demo:
            
            # Ana başlık
            gr.HTML("""
            <div class="main-header">
                <h1>🎨 Art Caption Generator</h1>
                <p>Analyze and describe your artworks with fine-tuned BLIP model</p>
            </div>
            """)
            
            with gr.Tab("🖼️ Single Image Caption"):
                gr.Markdown("### Generate captions for single image")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="🎨 Upload Artwork",
                            type="pil",
                            height=400,
                            sources=["upload", "webcam", "clipboard"]  # Enable paste from clipboard
                        )
                        
                        generation_mode = gr.Radio(
                            choices=["Conservative", "Balanced", "Creative"],
                            value="Balanced",
                            label="🎯 Generation Mode"
                        )
                        
                        creativity_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="🌟 Creativity Level (Creative mode only)"
                        )
                        
                        generate_btn = gr.Button("📝 Generate Caption", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        caption_output = gr.Markdown(
                            label="📝 Generated Caption",
                            value="Caption will appear here..."
                        )
                
                # Button click event
                generate_btn.click(
                    fn=self.generate_caption,
                    inputs=[image_input, generation_mode, creativity_slider],
                    outputs=caption_output
                )
            
            with gr.Tab("🔍 Advanced Analysis"):
                gr.Markdown("### Detailed artwork analysis")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        analysis_image = gr.Image(
                            label="🎨 Artwork to Analyze",
                            type="pil",
                            height=400,
                            sources=["upload", "webcam", "clipboard"]  # Enable paste from clipboard
                        )
                        
                        analyze_btn = gr.Button("🔍 Detailed Analysis", variant="secondary", size="lg")
                    
                    with gr.Column(scale=1):
                        analysis_output = gr.Markdown(
                            label="🔍 Analysis Results",
                            value="Analysis results will appear here..."
                        )
                
                # Analiz button
                analyze_btn.click(
                    fn=self.analyze_art_features,
                    inputs=analysis_image,
                    outputs=analysis_output
                )
            
            with gr.Tab("ℹ️ Model Info"):
                model_info = f"""
                ## 🤖 Model Information
                
                **Model Type:** {"🎨 Fine-tuned Art Model" if self.is_fine_tuned else "🔘 Base BLIP Model"}
                **Base Model:** Salesforce/blip-image-captioning-base
                **Device:** {self.device.type.upper()}
                **Framework:** PyTorch + Transformers
                
                ## 🎯 Features
                
                - **Conservative Mode:** Reliable results with beam search
                - **Balanced Mode:** Hybrid approach (beam + sampling)
                - **Creative Mode:** Creative descriptions with high creativity
                - **Advanced Analysis:** Detailed analysis from multiple perspectives
                
                ## 📚 Usage Tips
                
                1. **Image Quality:** High-resolution images produce better results
                2. **Art Types:** Model supports various art types (oil painting, watercolor, etc.)
                3. **Generation Modes:** Try different modes to find the best result
                4. **Creativity:** Higher values give more unique but riskier results
                
                ## 🎨 Supported Art Terminology
                
                - **Styles:** Abstract, realistic, impressionist, expressionist
                - **Techniques:** Oil painting, watercolor, acrylic, brushstroke
                - **Composition:** Foreground, background, perspective, depth
                - **Colors:** Palette, vibrant, muted, contrast
                """
                
                gr.Markdown(model_info)
            
            # Örnek görseller
            with gr.Tab("🖼️ Sample Images"):
                gr.Markdown("### Test with sample artworks")
                
                # Eğer sample images klasörü varsa göster
                sample_dir = "data/ArtCap_Images_Dataset"
                if os.path.exists(sample_dir):
                    sample_images = []
                    for img_file in os.listdir(sample_dir)[:8]:  # İlk 8 örnek
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(sample_dir, img_file)
                            try:
                                sample_images.append(img_path)
                            except:
                                continue
                    
                    if sample_images:
                        gr.Gallery(
                            value=sample_images[:6],
                            label="🎨 Sample Artworks",
                            show_label=True,
                            elem_id="gallery",
                            columns=3,
                            rows=2,
                            height="auto"
                        )
                
                gr.Markdown("""
                **Note:** You can drag and drop sample images to any of the tabs above to test them.
                **Tip:** Use Ctrl+V (or Cmd+V on Mac) to paste images from clipboard directly.
                """)
        
        return demo


def main():
    """Start demo"""
    print("🚀 Art Caption Demo starting...")
    
    # Check model path - handle both src/ and art_captioning/ execution
    import os
    current_dir = os.getcwd()
    
    # Possible model paths
    possible_paths = [
        "outputs/best_model.pth",           # From art_captioning/ 
        "../outputs/best_model.pth",       # From art_captioning/src/
        "../../outputs/best_model.pth"     # From other subdirs
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found fine-tuned model: {os.path.abspath(path)}")
            break
    
    if model_path is None:
        print("⚠️ Fine-tuned model not found, using standard BLIP")
        print(f"   Current directory: {current_dir}")
        print(f"   Searched paths: {possible_paths}")
    
    # Create demo
    demo_app = ArtCaptionDemo(model_path)
    interface = demo_app.create_interface()
    
    # Start interface
    print("🌐 Starting Gradio interface...")
    print("💡 Tip: You can paste images from clipboard using Ctrl+V (Cmd+V on Mac)")
    interface.launch(
        server_name="127.0.0.1",  # Localhost only for security
        server_port=7860,          # Port
        share=True,               # No public link
        debug=False,               # Less verbose output
        show_error=True,           # Show errors
        inbrowser=True            # Auto-open in browser
    )


if __name__ == "__main__":
    main() 