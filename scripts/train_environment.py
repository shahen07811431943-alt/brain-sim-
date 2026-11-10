import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain
from brain.environment import GridWorld
from brain.sleep import SleepEngine

print("Starting Interactive Environment Training...")

env = GridWorld(size=5)
brain = Brain(input_dim=10) # 5 خلايا لـ X و 5 لـ Y
sleep_engine = SleepEngine(brain.memory)

episodes = 50
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 1. الدماغ يقرر الحركة
        action = brain.act(state)
        
        # 2. البيئة تعطي النتيجة (الحالة الجديدة، المكافأة، هل انتهى؟)
        next_state, reward, done = env.step(action)
        
        # 3. الدماغ يتعلم من المكافأة (Three-Factor Learning)
        brain.process(next_state, reward=reward, label=f"ep_{ep}")
        
        state = next_state
        total_reward += reward

    if ep % 10 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}")

    # 4. النوم لتثبيت الذاكرة كل 10 جولات
    if ep % 10 == 0 and ep > 0:
        print("Sleeping to consolidate navigation memories...")
        sleep_engine.run_sleep_cycle(brain.network, replay_epochs=20)

print("Training complete. The Brain has learned to navigate the world!")
