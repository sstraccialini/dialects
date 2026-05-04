"""Central evaluation package."""

from .evaluation import run_evaluation, run_sentence_evaluation
from .parallel_alignment import run_parallel_alignment
from .compare_methods import run_cross_method_comparison

__all__ = [
    "run_evaluation",
    "run_sentence_evaluation",
    "run_parallel_alignment",
    "run_cross_method_comparison",
]
