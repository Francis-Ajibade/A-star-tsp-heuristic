# 🧠 A* Search for the Traveling Salesman Problem (TSP)

### Designing, Implementing, and Evaluating Heuristics

**Artificial Intelligence — Search and Heuristics**
**Author:** Francis Oladimeji Ajibade
**Course:** CS — Artificial Intelligence
**Year:** 2026

---

## 📌 Overview

This project investigates how **heuristic quality affects the performance of A* search** when applied to the **Traveling Salesman Problem (TSP)**.

Three heuristics were implemented and evaluated:

* **Null heuristic** (Uniform Cost Search baseline)
* **Minimum Edge heuristic**
* **Minimum Spanning Tree (MST) heuristic**

The central finding is:

> **Heuristic quality has an exponential—not marginal—impact on A* performance.**

---

## 🎯 Key Results

* MST reduced nodes expanded by:

  * **65% (5 cities)**
  * **92% (8 cities)**
  * **99%+ (10 cities)**

* On the real dataset (**TSPLIB95 – wi29**):

  * MST solved optimally in **268 ms (2,166 nodes)**
  * Min-edge required **53,227 nodes**
  * Null returned a **suboptimal solution**

---

## 🧠 Core Concepts

* A* Search Algorithm
* Heuristic Functions ( h(n) )
* Admissibility & Consistency
* NP-hard Problems
* State Space Search

---

## ⚙️ Problem Description

### 🟢 Simple Explanation

You must visit a set of cities and return home while minimizing total distance.

* Small cases → brute force works
* Large cases → combinations explode exponentially

This is why TSP is **NP-hard**.

---

### 🔵 Formal Definition

* Input: Distance matrix ( D[i][j] )
* Goal: Find shortest tour visiting all cities exactly once

A* evaluates states using:

[
f(n) = g(n) + h(n)
]

Where:

* `g(n)` = cost so far
* `h(n)` = estimated remaining cost

---

## 🔍 Heuristics Implemented

### 🔹 Null Heuristic

* ( h(n) = 0 )
* Equivalent to Uniform Cost Search
* Explores many unnecessary nodes

---

### 🔹 Minimum Edge Heuristic

* Uses global minimum edge as a lower bound
* Admissible but weak

---

### 🔹 MST Heuristic (Primary Contribution)

* Builds Minimum Spanning Tree over unvisited cities
* Adds entry + return edges
* Tight lower bound → **high efficiency**

---

## 📊 Experimental Results

### Random Instances

| Cities | Null | Min-edge | MST | Improvement |
| ------ | ---- | -------- | --- | ----------- |
| 5      | 41   | 28       | 14  | 65.9%       |
| 8      | 5043 | 3271     | 401 | 92.1%       |
| 10     | 89k+ | 12,145   | 25  | 99.97%      |

---

### 🌍 Real Dataset (wi29 – Western Sahara)

| Heuristic | Cost  | Nodes  | Time  | Optimal |
| --------- | ----- | ------ | ----- | ------- |
| Null      | 35.65 | 50,538 | 2.2s  | ❌       |
| Min-edge  | 30.88 | 53,227 | 4.6s  | ✅       |
| MST       | 30.88 | 2,166  | 268ms | ✅       |

---

## 🚨 Key Insight

* Weak heuristic → explores wrong paths
* Strong heuristic → prunes search space

> ❗ A* can return incorrect solutions if guidance is poor in practice

---

## ❗ Limitation

Even with MST:

* A* remains **exponential**
* Practical limit ≈ **12–15 cities**

This confirms:

> **TSP is NP-hard — no heuristic removes exponential growth**

---

## 🧱 Project Structure

```bash
A-star-tsp-heuristic/
│
├── code/
│   ├── phase1_foundation/
│   ├── phase2_astar/
│   ├── phase3_heuristics/
│
├── experiments/
│   ├── notebooks/
│
├── data/
│   └── wi29.tsp
│
├── results/
│   └── charts/
│
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/Francis-Ajibade/A-star-tsp-heuristic
cd A-star-tsp-heuristic
pip install -r requirements.txt
python code/phase2_astar/astar.py
```

---

## 🧠 Contributions

* Full A* implementation from scratch
* Verified admissibility & consistency checker
* Experimental benchmarking framework
* Real-world dataset validation

---

## 🔮 Future Work

* IDA* (memory-efficient A*)
* Dynamic heuristic selection
* Hybrid exact + approximation systems
* Machine learning-based heuristics

---

## 📚 References

* Held & Karp (1962) — Dynamic Programming
* TSPLIB95 — Benchmark dataset
* Russell & Norvig — A* theory
* Christofides — Approximation algorithm

---

## 👤 Author

**Francis Oladimeji Ajibade**
Computer Science — University of New Brunswick

---

## 🔗 Repository

👉 https://github.com/Francis-Ajibade/A-star-tsp-heuristic
