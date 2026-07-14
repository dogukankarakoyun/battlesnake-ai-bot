# Battlesnake AI Bot

A hybrid AI agent for the Battlesnake game combining heuristic strategies, pathfinding, and machine learning.

## Overview

This project implements an intelligent Battlesnake agent that combines multiple decision-making approaches:

- Heatmap-based evaluation (space and risk analysis)
- A* pathfinding for goal-oriented movement
- Machine learning predictions using LightGBM
- Final safety validation to avoid collisions

The bot dynamically combines these strategies to make stable and effective decisions during gameplay.

---

## Architecture

Decision pipeline:

1. Extract features from the current game state
2. Apply heatmap scoring
3. Use A* pathfinding as fallback
4. Use ML prediction as fallback
5. Apply final safety validation

---

## Tech Stack

- Python
- Flask
- LightGBM
- scikit-learn
- NumPy
- pandas
- A* pathfinding
- Flood-fill algorithm

---

## Project Structure

```text
src/
├── Battlesnake/
│   ├── astar/
│   ├── game.py
│   ├── heatmap.py
│   ├── main.py
│   ├── path_fallback.py
│   ├── server.py
│   ├── strategy.py
│   └── utils.py
│
├── LightGBM/
│   ├── training/
│   ├── ml_features.py
│   ├── run.py
│   ├── insane_model.pkl
│   └── feature_importance.png
│
docs/
└── screenshots/
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the bot:

```bash
python src/Battlesnake/main.py
```

The server runs at:

```text
http://localhost:8000
```

---

## Model Training (Optional)

To train the LightGBM model:

```bash
python src/LightGBM/training/train_model.py
```

---

## Features

- Hybrid AI architecture
- Dynamic strategy switching
- Heatmap-based board evaluation
- A* pathfinding
- Flood-fill space analysis
- ML-assisted move prediction
- Collision prevention and safety checks

---

## Notes

This project was developed as part of a university team project.
