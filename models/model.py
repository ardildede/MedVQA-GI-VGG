import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG_Attention_VQA(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=512, hidden_dim=512):
        super(VGG_Attention_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: VGG19 (Spatial Features)
        # VGG'nin sadece özellik çıkarıcı kısmını alıyoruz (FC katmanları yok)
        # Çıktı: [Batch, 512, 7, 7] (224x224 resim için)
        print("Model: VGG19 Yükleniyor...")
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features 
        
        # 2. METİN: LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. ATTENTION (DİKKAT) KATMANLARI
        # Resimdeki her bölge (v) ve soru (q) için ortak bir uzay yaratıyoruz
        self.v_proj = nn.Linear(512, hidden_dim) # Resmi projeksiyonla
        self.q_proj = nn.Linear(hidden_dim, hidden_dim) # Soruyu projeksiyonla
        self.w_att = nn.Linear(hidden_dim, 1) # Puanla
        
        # 4. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, questions):
        # --- A. Resim Özellikleri ---
        # [Batch, 512, 7, 7]
        img_feat = self.features(images)
        b, c, h, w = img_feat.size()
        
        # Düzleştir: [Batch, 49, 512] (49 tane bölge)
        img_feat = img_feat.view(b, c, -1).permute(0, 2, 1)
        
        # --- B. Soru Özellikleri ---
        embeds = self.embedding(questions)
        # LSTM'den son gizli durumu (hidden state) al: [Batch, 512]
        _, (hidden, _) = self.lstm(embeds)
        ques_feat = hidden[-1]
        
        # --- C. DİKKAT MEKANİZMASI (ATTENTION) ---
        # 1. Resimdeki 49 bölgeyi projeksiyonla: [Batch, 49, 512]
        v_proj = self.v_proj(img_feat) 
        
        # 2. Soruyu 49 kere tekrarla (her bölgeyle kıyaslamak için): [Batch, 49, 512]
        q_proj = self.q_proj(ques_feat).unsqueeze(1).expand_as(v_proj)
        
        # 3. Birleştir ve Tanh -> Linear -> Softmax
        # Formül: alpha = softmax(w * tanh(v + q))
        attention_scores = self.w_att(torch.tanh(v_proj + q_proj)) # [Batch, 49, 1]
        alpha = F.softmax(attention_scores, dim=1) # Dikkat ağırlıkları (Nereye bakmalı?)
        
        # 4. Ağırlıklı Resim Vektörü
        # Her bölgeyi kendi dikkat puanıyla çarpıp topluyoruz
        # Polip olan yerin puanı yüksekse orası parlar.
        weighted_img = (img_feat * alpha).sum(dim=1) # [Batch, 512]
        
        # --- D. Sınıflandırma ---
        # Dikkatle seçilmiş resim bilgisi + Soru bilgisi
        combined = weighted_img + ques_feat
        output = self.classifier(combined)
        
        return output