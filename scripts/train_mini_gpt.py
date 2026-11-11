import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wikipediaapi
import numpy as np

print("🧠 Building Mini-GPT (Generative Neural Network) from Scratch...")

# 1. جلب البيانات
wiki = wikipediaapi.Wikipedia(user_agent='MiniGPT/1.0', language='ar')
topics = ["الذكاء_الاصطناعي", "الدماغ_البشري", "الفيزياء", "الفضاء"]
text = ""
for t in topics:
    page = wiki.page(t)
    if page.exists():
        text += page.summary + " "

# 2. بناء القاموس (Vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# 3. تحويل النص إلى أرقام
encoded = [char_to_int[ch] for ch in text]

# 4. بناء الشبكة العصبية (Mini-GPT)
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=1):
        super(MiniGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

model = MiniGPT(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 5. التدريب (Learning to Generate)
print("⚡ Training the Neural Network (Generating Text)...")
seq_length = 20
epochs = 5 # سنكتفي بـ 5 دورات لسرعة التنفيذ

for epoch in range(epochs):
    for i in range(0, len(encoded) - seq_length, seq_length):
        inputs = torch.tensor(encoded[i:i+seq_length]).unsqueeze(0)
        targets = torch.tensor(encoded[i+1:i+seq_length+1]).squeeze(0)
        
        optimizer.zero_grad()
        output, _ = model(inputs)
        loss = criterion(output.squeeze(0), targets)
        loss.backward()
        optimizer.step()
    print(f"   Epoch {epoch+1}/{epochs} completed. Loss: {loss.item():.4f}")

# 6. حفظ الدماغ (النموذج والقاموس)
torch.save({
    'model_state': model.state_dict(),
    'char_to_int': char_to_int,
    'int_to_char': int_to_char
}, "mini_gpt.pth")

print("💾 Brain saved to mini_gpt.pth! It is now a Generative Model.")
