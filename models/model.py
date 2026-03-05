import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        # Hangi özelliğin daha önemli olduğunu seçen kapı katmanı
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Özellikleri birleştir ve kapıdan geçir
        combined = torch.cat((x1, x2), dim=1)
        gate_values = self.gate(combined)
        return combined * gate_values

class VGG_BERT_Gated_VQA(nn.Module):
    def __init__(self, num_classes):
        super(VGG_BERT_Gated_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: VGG16 (Pre-trained)
        vgg = models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        self.vgg_avgpool = vgg.avgpool
        # VGG çıktısını 768 boyuta indir (BERT ile eşitlemek için)
        self.vgg_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 768),
            nn.ReLU()
        )
        
        # 2. METİN: DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 3. FUSION: Gated Fusion (768 + 768 = 1536)
        self.fusion = GatedFusion(input_dim=1536)
        
        # 4. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Görüntü Özellikleri
        img_x = self.vgg_features(images)
        img_x = self.vgg_avgpool(img_x)
        img_features = self.vgg_fc(img_x)
        
        # Metin Özellikleri (BERT CLS Token)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_out.last_hidden_state[:, 0, :] # [CLS] token
        
        # Gated Fusion
        fused = self.fusion(img_features, text_features)
        
        # Sınıflandırma
        return self.classifier(fused)