"""
Task-conditioning modules.

This package contains utilities that turn *task information* (IDs, language prompts, etc.)
into token sequences suitable for attention-based fusion modules.
"""

from policies.modules.task.learnable_query import LearnableQueryTokens
from policies.modules.task.language_query import LanguageQueryTokens

__all__ = [
    "LearnableQueryTokens",
    "LanguageQueryTokens",
]

