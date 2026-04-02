"""
Flow-matching testbed.

Contains a small, readable framework for running sanity-check experiments
against `policies.algorithms.flow_matching.FlowMatchingAlgorithm`.
"""

from policies.testing.flow_matching.framework import ExperimentConfig, FlowMatchingExperiment

__all__ = [
    "ExperimentConfig",
    "FlowMatchingExperiment",
]

