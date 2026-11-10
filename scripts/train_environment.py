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

episodes = 100 # زيادة الجولات ليتعلم أكثر
max_steps = 50 

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # تقليل الاستكشاف تدريجياً (من 0.8 إلى 0.05)
    epsilon = max(0.05, 0.8 - (ep / episodes) * 0.75)
    
    while not done and steps < max_steps:
        action = brain.act(state, epsilon=epsilon)
        next_state, reward, done = env.step(action)
        
        # الدماغ يتعلم من التجربة (State, Action, Reward, Next State)
        brain.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        steps += 1

    # طباعة النتيجة كل 10 جولات
    if ep % 10 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.2f}")

    # النوم كل 20 جولة لتثبيت الذاكرة
    if ep % 20 == 0 and ep > 0:
        print("Sleeping to consolidate navigation memories...")
        sleep_engine.run_sleep_cycle(brain.network, replay_epochs=20)

print("Training complete. The Brain has learned to navigate the world!")
