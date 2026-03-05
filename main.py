%%writefile vqa_project/main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Importlar
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import VGG_Attention_VQA

# --- TOKENIZER FONKSİYONU ---
def tokenize_batch(questions, vocab, max_len=20):
    batch_indices = []
    for q in questions:
        tokens = q.lower().split()
        indices = [vocab.get(t, 0) for t in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        batch_indices.append(indices)
    return torch.tensor(batch_indices, dtype=torch.long)

# --- GRAFİK ÇİZME FONKSİYONU ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('VGG + Attention Confusion Matrix')
    
    plt.savefig("vgg_matrix.png", dpi=300)
    print("📊 Karmaşıklık matrisi 'vgg_matrix.png' olarak kaydedildi.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- DİKKAT MEKANİZMASI (VGG + Attention) Başlatılıyor ---")
    print(f"Kullanılan Cihaz: {device}")

    # 1. VERİ ve SÖZLÜK
    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data)
    
    print("Sözlükler oluşturuluyor...")
    answers_list = train_data['answer']
    all_answers = sorted(list(set(str(ans).lower() for ans in answers_list)))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}
    
    questions_list = train_data['question']
    all_text = " ".join([str(q).lower() for q in questions_list])
    unique_words = set(all_text.split())
    vocab = {word: i+1 for i, word in enumerate(unique_words)}
    vocab['<unk>'] = 0 
    vocab['<pad>'] = 0 
    
    print(f"Sınıf Sayısı: {len(answer_map)}")
    print(f"Kelime Dağarcığı: {len(vocab)}")

    # 2. DATASET
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = KvasirHFDataset(train_data, answer_map, transform=transform)
    val_ds = KvasirHFDataset(val_data, answer_map, transform=transform)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['answer'] for item in batch])
        raw_questions = [item['question'] for item in batch]
        question_tensors = tokenize_batch(raw_questions, vocab)
        return images, question_tensors, labels

    # VGG ağırdır, Kaggle T4 için 32 uygundur, lokalde (4GB VRAM) 8 kullan.
    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 3. MODEL
    model = VGG_Attention_VQA(
        num_classes=len(answer_map), 
        vocab_size=len(vocab)+1
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- AKILLI KAYDETME SİSTEMİ ---
    MODEL_PATH = "vgg_attn_model.pth"

    if os.path.exists(MODEL_PATH):
        print(f"\n✅ Kayıtlı model bulundu ({MODEL_PATH}). Eğitim ATLANIYOR...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except:
            model.load_state_dict(torch.load(MODEL_PATH), strict=False)
            print("⚠️ Model esnek modda yüklendi.")
    else:
        print("\n--- Eğitim Başlıyor ---")
        EPOCHS = 3
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for imgs, quests, labels in loop:
                imgs, quests, labels = imgs.to(device), quests.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs, quests)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} Bitti. Loss: {total_loss/len(train_loader):.4f}")
        
        # Eğitimi Kaydet
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Model başarıyla kaydedildi: {MODEL_PATH}")

    # 4. TEST VE RAPORLAMA
    print("\n--- Test Raporu Hazırlanıyor ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, quests, labels in tqdm(val_loader):
            imgs, quests, labels = imgs.to(device), quests.to(device), labels.to(device)
            
            outputs = model(imgs, quests)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Listeleri Hazırla
    label_ids = list(answer_map.values())
    label_names = list(answer_map.keys())

    # A) TXT RAPORU
    report_text = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0)
    with open("vgg_rapor.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("✅ 'vgg_rapor.txt' kaydedildi.")

    # B) EXCEL/CSV RAPORU
    report_dict = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0,
                                        output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv("vgg_analiz.csv")
    print("✅ 'vgg_analiz.csv' kaydedildi (Excel formatı).")
    
    # C) GRAFİK
    plot_confusion_matrix(all_labels, all_preds, label_names)
    
    print("\n🎉 Tüm işlemler tamamlandı. Output klasörünü kontrol et!")

if __name__ == "__main__":
    main()