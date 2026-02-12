"""Provider profiles for the Coding Agent Loop.

Profiles configure how the agent behaves differently for each LLM provider:
system prompts, tool descriptions, behavioral defaults.
"""

from attractor_agent.profiles.base import BaseProfile, ProviderProfile
from attractor_agent.profiles.registry import get_profile, list_profiles

__all__ = [
    "ProviderProfile",
    "BaseProfile",
    "get_profile",
    "list_profiles",
]
