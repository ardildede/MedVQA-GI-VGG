import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Proje İçi Importlar
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import VGG_BERT_Gated_VQA # Yeni model sınıfı ismi

# --- GRAFİK ÇİZME FONKSİYONU ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('VGG + BERT Gated Fusion Confusion Matrix')
    plt.savefig("vgg_bert_matrix.png", dpi=300)
    print("📊 Karmaşıklık matrisi 'vgg_bert_matrix.png' olarak kaydedildi.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- VGG + BERT Gated Fusion Modeli Başlatılıyor ---")
    print(f"Kullanılan Cihaz: {device}")

    # 1. HAZIRLIK (Tokenizer ve Veri)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data)
    
    # Cevap haritasını oluştur (Sınıflandırma için)
    answers_list = train_data['answer']
    all_answers = sorted(list(set(str(ans).lower() for ans in answers_list)))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}
    label_names = list(answer_map.keys())

    # 2. DATASET VE TRANSFORM
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = KvasirHFDataset(train_data, answer_map, transform=transform)
    val_ds = KvasirHFDataset(val_data, answer_map, transform=transform)

    # BERT ve Görüntü verilerini birleştiren Collate fonksiyonu
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['answer'] for item in batch])
        
        # Metinleri BERT formatına çevir
        questions = [item['question'] for item in batch]
        encoded_text = tokenizer(
            questions, 
            padding=True, 
            truncation=True, 
            max_length=32, 
            return_tensors="pt"
        )
        
        return images, encoded_text['input_ids'], encoded_text['attention_mask'], labels

    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 3. MODEL TANIMLAMA
    model = VGG_BERT_Gated_VQA(num_classes=len(answer_map)).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5) # BERT için AdamW daha iyidir
    criterion = nn.CrossEntropyLoss()

    MODEL_PATH = "vgg_bert_gated_model.pth"

    # EĞİTİM DÖNGÜSÜ
    if os.path.exists(MODEL_PATH):
        print(f"✅ Kayıtlı model bulundu: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("\n--- Eğitim Başlıyor (3 Epoch) ---")
        for epoch in range(3):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for imgs, ids, masks, lbs in loop:
                imgs, ids, masks, lbs = imgs.to(device), ids.to(device), masks.to(device), lbs.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs, ids, masks) # 3 Girdi gönderiyoruz
                loss = criterion(outputs, lbs)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), MODEL_PATH)

    # 4. TEST VE RAPORLAMA
    print("\n--- Test Aşaması ---")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, ids, masks, lbs in tqdm(val_loader):
            imgs, ids, masks, lbs = imgs.to(device), ids.to(device), masks.to(device), lbs.to(device)
            outputs = model(imgs, ids, masks)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

    # Raporları Kaydet
    report_text = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    with open("vgg_bert_rapor.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    plot_confusion_matrix(all_labels, all_preds, label_names)
    print("🎉 İşlem başarıyla tamamlandı!")

if __name__ == "__main__":
    main()