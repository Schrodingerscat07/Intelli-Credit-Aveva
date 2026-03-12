# Intelli-Credit-Aveva — Industrial AI Optimization Engine

> **Hackathon Prototype** — IITH x Aveva
> A full-stack AI platform built to solve the ultimate manufacturing dilemma: maximizing product quality and yield while strictly minimizing energy consumption and carbon emissions.

---

## 🏭 The Problem: Energy vs. Quality Trade-off

Imagine a massive factory producing millions of medicine tablets. The machines have hundreds of dials and settings (like temperature, pressure, rotation speed). The factory managers are playing a constant game of tug-of-war:

1. They want to make the **best quality tablets** as fast as possible to maximize yield.
2. They want to use the **least amount of electricity** possible to hit Carbon footprint targets and save money.

Usually, if you crank up the machine speed to make more tablets, you drastically increase power usage or produce damaged (friable) tablets. Finding the "perfect balance" (the optimal machine settings) given the live telemetry of the factory is nearly impossible for a human to do in real-time.

---

## 🤖 Our Solution: The AI Supervisor

Our project is an **AI-driven supervisor** that finds this perfect balance automatically.

1. **It learns from history:** It ingests massive amounts of past factory data and simulates millions of scenarios to locate the absolute best machine "recipes" (we call them Golden Signatures), where energy is lowest but quality remains high.
2. **It operates in real-time:** It monitors live machines and uses a lightning-fast PyTorch neural network to dynamically suggest new settings.
3. **It respects human authority:** Before any machine dials are turned, the AI pauses and asks an operator for approval using a Human-In-The-Loop (HITL) system.
4. **It learns continuously:** Every time a new "recipe" works better than the last, it saves it. The system literally gets smarter every single batch.

---

## 🎯 Hackathon Problem Statement Alignment

We have chosen **Option B: Optimization Engine Track**. Our project explicitly fulfills all core and universal objectives outlined in the case:

- ✅ **Golden Signature Framework:** We engineered a custom NSGA-II algorithm to discover Pareto-optimal settings, storing them in a Qdrant Vector database for live comparison.
- ✅ **Continuous Learning:** Our system automatically tracks batch outcomes and upserts new performance benchmarks (signatures) into memory when historical results are beaten.
- ✅ **Human-in-the-Loop workflows (HITL):** Using a LangGraph state machine, execution dynamically interrupts to allow humans to review and accept/reject proposed changes via an API-driven React dashboard.
- ✅ **Multi-Target Optimization:** Our intelligent proxy model inherently balances the primary (Tablet Weight, Friability, Hardness) against the secondary targets (Power Consumption).

---

## 📂 Project Structure

This repository is split into two primary environments:

### 1. The Backend (`/backend`)
The data engineering, machine learning engine, LangGraph orchestration, and API layer.
👉 [Read the detailed Technical Backend Architecture here](./backend/README.md)

### 2. The Frontend (`/frontend`)
The React + Vite dashboard displaying the current batch status, historical comparisons, and the human-operator approval gates.
👉 [Read the Frontend documentation here](./frontend/README.md)

---

## 🚀 Quick Execution Guide

For comprehensive setup, please refer to the specific folder READMEs. 

To run the full stack:
1. Navigate to `/backend` and run the core engine to generate the models and training sets:
   `python offline_optimizer.py` followed by `python orchestration_layer.py`.
2. Open a new terminal, navigate to `/frontend`, and start the dashboard with `npm run dev`.

---

## 📄 License
This prototype is released under the [LICENSE](LICENSE).