# pyOptiLab

pyOptiLab is a personal learning project where I explore combinatorial optimization and mathematical programming problems by implementing different algorithms in Python.

The goal of this repository is to improve my programming skills while learning more about optimization techniques, algorithm design, and benchmarking.

Current and planned problems include:

- Knapsack
- Graph Coloring
- Traveling Salesman Problem (TSP)
- Scheduling problems

## Running

This project uses **uv** for dependency and environment management.

Install dependencies:

```bash
uv sync
```

Run the CLI:

```bash
uv run pyoptilab --help
```

For each problem, three different commands are exposed in general

- generate : generates instances from a certain distribution and exports them to a json file

```bash
uv run pyoptilab <PROBLEM> generate --help
```

- solve : solves instances extracted from json instance files with specific solvers

```bash
uv run pyoptilab <PROBLEM> solve --help
```

- run : generates instances from a certain distribution and directly solves them with specific solvers

```bash
uv run pyoptilab <PROBLEM> run --help
```

## Status

This is an active personal project and is expected to evolve as I explore new optimization problems and algorithms.
