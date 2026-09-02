# Coordinated Abuse Risk Manager

An event-driven risk detection system exploring whether coordinated abuse can be identified from patterns that are difficult to capture at the individual-transaction level.

The project will build three detectors over the same chronological transaction stream:

- **Baseline A — Transaction-level:** uses transaction-local information.
- **Baseline B — Static graph:** uses transaction-local information plus static infrastructure relationships.
- **Temporal detector:** uses transaction-local information plus historical graph and evolving behavioral context.

The system is designed around two independently generated synthetic worlds:

- **World A:** learning/development environment used for model training, validation, and threshold selection.
- **World B:** held-out deployment environment containing new entities and a new realization of the underlying abuse mechanisms.

Both worlds will stream events through Kafka. Neo4j will maintain graph state used by the graph-based detectors. Models will be evaluated using the information available to them at the time each event arrives.

## Initial scope

The implementation will focus on:

- chronological event generation and streaming;
- explicit information boundaries between detectors;
- static infrastructure relationships;
- evolving behavioral relationships;
- temporal feature computation using historical state;
- frozen-model deployment against an independently generated held-out world;
- evaluation of detection quality, timing, false positives, and exposure-related outcomes.

The exact data-generating parameters, model results, thresholds, and performance claims will be established through experiments and documented after implementation.

## Planned stack

- Python
- Apache Kafka
- Neo4j
- LightGBM
- scikit-learn
- pandas / NumPy
- PyYAML
- Pydantic

## Repository status

Early-stage implementation. This initial commit contains the project skeleton and starting documentation.
