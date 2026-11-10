# 🧠 Brain-Sim: The Syncretic E-Brain Architecture

**Brain-Sim** is an open-source, biologically-inspired cognitive architecture that simulates the human brain's learning mechanisms. Unlike traditional deep learning models that rely on massive datasets and suffer from catastrophic forgetting, Brain-Sim learns from interaction, builds spatial maps, and prevents forgetting through a simulated **Wake-Sleep cycle**.

## 🚀 Key Features
- **Spiking Neural Network (SNN)**: Basic Leaky Integrate-and-Fire (LIF) neurons with local STDP.
- **Three-Factor Learning (R-STDP)**: Incorporates Dopamine-like reward signals for reinforcement learning.
- **Complementary Learning Systems**: A dual-memory system (Hippocampus for fast encoding, Cortex for slow consolidation).
- **Sleep Replay**: Simulated "Sleep" phases to prevent Catastrophic Forgetting (tested and proven).
- **Novelty Detection (VIBE-inspired)**: Autonomous detection of new patterns for attention allocation.
- **Spatial Generalization**: Q-Learning based cognitive mapping that adapts to dynamic goals in environments with obstacles.

## 📊 Experimental Results
Our tests demonstrate the system's ability to master complex environments:
- **Catastrophic Forgetting Prevention**: After learning Task A and Task B, the Brain retains 100% accuracy on Task A after a Sleep Cycle.
- **Dynamic Environment Mastery**: In a 5x5 GridWorld with obstacles and a *moving goal*, the Brain learns to find the optimal path (7-8 steps) by building a spatial map, not just memorizing routes.

## 🛠️ Quick Start

```bash
git clone https://github.com/shahen07811431943-alt/brain-sim-.git
cd brain-sim-
pip install -r requirements.txt

# Run Wake Phase Training (Environment & Plasticity Tests)
python scripts/train_wake.py
python scripts/train_environment.py

# Run Sleep Phase (Memory Consolidation)
python scripts/run_sleep_cycle.py
