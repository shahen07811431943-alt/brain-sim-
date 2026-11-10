import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain
from brain.environment import GridWorld
from brain.sleep import SleepEngine

print("Starting Interactive Environment Training...")

env = GridWorld(size=5)
brain = Brain(input_dim=10) 
sleep_engine = SleepEngine(brain.memory)

episodes = 50
max_steps = 50 # الحد الأقصى للحركات في الجولة الواحدة لمنع التعليق

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # أضفنا شرط الخطوات هنا
    while not done and steps < max_steps:
        action = brain.act(state)
        next_state, reward, done = env.step(action)
        brain.process(next_state, reward=reward, label=f"ep_{ep}")
        state = next_state
        total_reward += reward
        steps += 1 # زيادة عداد الخطوات

    # إذا انتهت الخطوات ولم يصل، نعتبر الجولة منتهية
    if not done:
        done = True 

    if ep % 10 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Steps = {steps}")

    if ep % 10 == 0 and ep > 0:
        print("Sleeping to consolidate navigation memories...")
        sleep_engine.run_sleep_cycle(brain.network, replay_epochs=20)

print("Training complete. The Brain has learned to navigate the world!")
