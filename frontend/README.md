# Intelli-Credit-Aveva Application Frontend

This frontend is a React + Vite dashboard designed to provide a Human-In-The-Loop (HITL) interface for the Intelli-Credit-Aveva optimization engine.

## Overview

The AI engine constantly optimizes machine settings to balance tablet yield and energy consumption. However, before radical new "Golden Signatures" (machine settings) are applied to the real factory floor, they require human approval. This React dashboard provides the UI to review, accept, or reject these changes.

## Features

- **Real-Time Data Visualization**: Displays current batch performance against historical golden signatures.
- **Human-In-The-Loop Approval Gates**: Surfaces proposed optimizations (e.g., changes to machine speed, pressure, temperature) and their predicted impact on energy usage and yield.
- **Continuous Improvement Integration**: Allows operators to reprioritize targets and feedback directly impacts the AI's continuous learning cycle.

## Integration points

The dashboard communicates with the python backend APIs:
- `GET /api/pending-approval`: Polls for any optimizations paused in the LangGraph state machine waiting for human review.
- `POST /api/approve`: Sends the operator's decision (`approved: True/False`) back to the LangGraph orchestration layer, resuming the execution node.

## Getting Started

### Prerequisites

- Node.js installed

### Installation & Running

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The React Compiler is not enabled on this current Vite plugin React setup to preserve fast dev processing. We recommend using ESLint to maintain component architecture.
