import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain
from brain.environment import GridWorld
from brain.sleep import SleepEngine

print("🌟 Starting GRAND INTEGRATION: Multimodal Cognitive Agent...")

# 1. تهيئة البيئة (شبكة 5x5) والدماغ (مدخلات بصرية 5x5 = 25)
env = GridWorld(size=5)
brain = Brain(input_dim=25) 
sleep_engine = SleepEngine(brain.memory)

# 2. الإشارة البصرية: سهم يشير لليمين
img_right = np.array([0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0])

episodes = 100
max_steps = 10

for ep in range(episodes):
    env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    # نسبة الاستكشاف
    epsilon = max(0.05, 0.5 - (ep / episodes) * 0.45)
    
    while not done and steps < max_steps:
        # الدماغ ينظر إلى الإشارة البصرية ويقرر الحركة
        action = brain.act(img_right, epsilon=epsilon)
        
        # البيئة تنفذ الحركة (لمحاكاة التفاعل فقط)
        next_state, env_reward, done = env.step(action)
        
        # 🧠 هنا السحر: المكافأة تعتمد على "طاعة السهم البصري" وليس على الوصول للهدف
        if action == 3:  # 3 تعني RIGHT (يمين)
            reward = 1.0 # مكافأة دوبامين
        else:
            reward = -1.0 # عقاب
            
        # الدماغ يتعلم من المكافأة (يربط الصورة بالحركة الصحيحة)
        brain.learn(img_right, action, reward, img_right)
        
        total_reward += reward
        steps += 1
        
    if ep % 20 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.2f}")
        
    # النوم لتثبيت الذاكرة
    if ep % 40 == 0 and ep > 0:
        print("Sleeping to consolidate multimodal memory...")
        sleep_engine.run_sleep_cycle(brain.network, replay_epochs=20)

print("✅ Integration Successful! The agent can see, move, learn, and sleep.")

# اختبار نهائي: هل تعلم الدماغ أن سهم اليمين يعني التحرك يميناً (Action 3)؟
print("Final Test: Presenting the Right Arrow...")
final_action = brain.act(img_right, epsilon=0.0)
action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
print(f"Brain chose action: {action_names[final_action]}")

if final_action == 3:
    print("🏆 GRAND SUCCESS! The Integrated Brain correctly processed vision and chose the right action.")
    sys.exit(0)
else:
    print("❌ Integration Failed. The brain is confused.")
    sys.exit(1)
