import torch
import torch.nn as nn

# (نفس تعريف الشبكة)
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

print("📬 Reading question and generating response...")
with open("question.txt", "r", encoding="utf-8") as f:
    question = f.read().strip()

# تحميل الدماغ
checkpoint = torch.load("mini_gpt.pth")
char_to_int = checkpoint['char_to_int']
int_to_char = checkpoint['int_to_char']
vocab_size = len(char_to_int)

model = MiniGPT(vocab_size)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# توليد النص
def generate_text(start_string, length=100):
    chars = [ch for ch in start_string if ch in char_to_int]
    if not chars: return "عذراً، لا أفهم هذه الحروف."
    
    input_seq = torch.tensor([char_to_int[ch] for ch in chars]).unsqueeze(0)
    hidden = None
    result = start_string
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            predicted_idx = torch.multinomial(probs, 1).item()
            predicted_char = int_to_char[predicted_idx]
            result += predicted_char
            input_seq = torch.tensor([[predicted_idx]])

    return result

# الدماغ يكمل الجملة
response = generate_text(f"سألتني: {question}. إجابتي هي: ", length=150)

with open("answer.txt", "w", encoding="utf-8") as f:
    f.write(response)

with open("chat_history.md", "a", encoding="utf-8") as f:
    f.write(f"**👤 أنت:** {question}\n\n**🧠 الدماغ:** {response}\n\n---\n")

print("✅ Brain generated a response! Check answer.txt")
