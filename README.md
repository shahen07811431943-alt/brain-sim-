# 🧠 Brain-Sim: A Syncretic Biological Cognitive Architecture

**Brain-Sim** is an open-source, biologically-inspired AI framework that simulates the human brain's learning mechanisms. Unlike traditional models that rely on massive datasets and suffer from catastrophic forgetting, Brain-Sim learns from interaction, builds spatial maps, and prevents forgetting through a simulated **Wake-Sleep cycle**.

Built as a fully continuous integration (CI/CD) pipeline on GitHub Actions, this project proves that complex, multimodal cognitive tasks can be achieved with lightweight, brain-inspired algorithms.

## 🚀 Key Features
- **Spiking Substrate & Local Plasticity**: Basic Leaky Integrate-and-Fire (LIF) neurons with Spike-Timing-Dependent Plasticity (STDP).
- **Three-Factor Learning (Dopamine)**: Integrates a reward signal (Q-Learning) to guide local plasticity towards goal-directed behavior.
- **Complementary Learning Systems**: A dual-memory system (Hippocampus for fast episodic encoding, Cortex for slow consolidation) to prevent Catastrophic Forgetting.
- **Sleep Replay Engine**: Simulated sleep phases to replay memories and consolidate them into long-term storage.
- **Novelty Detection**: Autonomous detection of new patterns to allocate learning resources (VIBE-inspired).
- **Visual Perception**: Learns to extract features and recognize patterns (e.g., noisy images) via Hebbian learning.
- **Predictive Coding**: Learns temporal sequences to predict the next stimulus.
- **Multimodal Binding**: Directly links visual perception to motor actions (Vision-to-Action).
- **Spatial Generalization**: Adapts to dynamic environments with obstacles and moving goals.

## 📊 Experimental Milestones (CI/CD Verified)
All tests run automatically on every push via GitHub Actions:
- ✅ **Catastrophic Forgetting**: Retains 100% accuracy on previous tasks after a sleep cycle.
- ✅ **Dynamic Navigation**: Finds the optimal path (8 steps) in a 5x5 GridWorld with obstacles and moving goals.
- ✅ **Visual Recognition**: Recognizes a noisy 'L' pattern with 100% confidence.
- ✅ **Predictive Coding**: Accurately predicts the next element in a temporal sequence.
- ✅ **Multimodal Agent**: Learns to associate a visual cue (Right Arrow) with the correct motor action (Move Right) in an interactive environment.

## 🛠️ Project Structure
- `.github/workflows/`: Automated CI/CD pipelines (`wake.yml`, `sleep.yml`).
- `brain/`: The core cognitive architecture.
  - `neuron.py`: LIF neurons.
  - `network.py`: Synaptic matrices.
  - `plasticity.py`: STDP and Three-Factor learning.
  - `memory.py`: Hippocampus & Cortex models.
  - `sleep.py`: Memory consolidation engine.
  - `novelty.py`: Dynamic threshold novelty detection.
  - `environment.py`: Interactive GridWorld.
  - `brain.py`: The central orchestrator.
- `scripts/`: Training and testing scripts for each cognitive module.
- `tests/`: Pytest suites for local plasticity.

## 🚀 Quick Start
```bash
git clone https://github.com/shahen07811431943-alt/brain-sim-.git
cd brain-sim-
pip install -r requirements.txt

# Run Wake Phase (Vision, Prediction, Navigation)
python scripts/train_integrated_brain.py

# Run Sleep Phase (Memory Consolidation)
python scripts/run_sleep_cycle.py
