import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain
from brain.environment import GridWorld
from brain.sleep import SleepEngine

print("Starting Interactive Environment Training with OBSTACLES...")

env = GridWorld(size=5)
brain = Brain(input_dim=10) 
sleep_engine = SleepEngine(brain.memory)

episodes = 300 # زيادة الجولات للتعلم وسط العقبات
max_steps = 100 

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # تقليل الاستكشاف ببطء (من 0.9 إلى 0.05)
    epsilon = max(0.05, 0.9 - (ep / episodes) * 0.85)
    
    while not done and steps < max_steps:
        action = brain.act(state, epsilon=epsilon)
        next_state, reward, done = env.step(action)
        
        brain.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        steps += 1

    # طباعة النتيجة كل 30 جولة
    if ep % 30 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.2f}")

    # النوم كل 50 جولة
    if ep % 50 == 0 and ep > 0:
        print("Sleeping to consolidate navigation memories...")
        sleep_engine.run_sleep_cycle(brain.network, replay_epochs=30)

print("Training complete. The Brain has mastered the obstacles!")
