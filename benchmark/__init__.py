"""Deterministic benchmark comparing NanoCode's normal and RLM paths.

This is an engineering benchmark, not a scientific evaluation: the task set is
tiny, the model is a scripted stand-in, and nothing here supports a claim of
statistical significance. It exists to measure what the two execution paths
actually do on the same task -- how many model calls, how many tool calls, how
many children, and whether the known bug was found.
"""
