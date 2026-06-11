"""Headless benchmark harness for the gameplay engine.

Drives the deterministic world engine (action-space + resolvers from
``agents.gameplay_node``) with pluggable, LLM-free policies so win-rate and
ticks-to-win can be measured over many saved worlds without any model calls.

This is the baseline-measurement layer for comparing policies
(random / heuristic / future RL) on the same set of generated rooms. It never
touches the live LLM gameplay path.
"""
