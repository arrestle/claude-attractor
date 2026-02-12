"""Profile registry: lookup profiles by provider name."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attractor_agent.profiles.base import ProviderProfile


def get_profile(provider: str) -> ProviderProfile:
    """Get the profile for a provider. Falls back to BaseProfile."""
    from attractor_agent.profiles.anthropic import AnthropicProfile
    from attractor_agent.profiles.base import BaseProfile
    from attractor_agent.profiles.gemini import GeminiProfile
    from attractor_agent.profiles.openai import OpenAIProfile

    registry: dict[str, ProviderProfile] = {
        "anthropic": AnthropicProfile(),
        "openai": OpenAIProfile(),
        "gemini": GeminiProfile(),
    }
    return registry.get(provider, BaseProfile())


def list_profiles() -> list[str]:
    """Return all registered profile names."""
    return ["anthropic", "openai", "gemini"]
