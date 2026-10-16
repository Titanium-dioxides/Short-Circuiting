"""Shared infrastructure for the Short-Circuiting project.

Contains the private-communication architecture (public residual writeback
deletion + private A->B Query/Value modulation), routing, adapters, hook
backend, generic interventions, checkpointing, statistics and reproducibility
utilities. Must NOT import from `questions.*` and must NOT hardcode IOI
names, templates, head ids or label semantics.
"""
