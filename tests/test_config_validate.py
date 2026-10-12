import pytest

from agent.config_validate import validate_config, ConfigError


def _cfg(**overrides):
    cfg = {
        "llm": {"api_key": "sk-real"},
        "agent": {"checkpoint": {"backend": "sqlite"}},
        "harness": {"guardrails": {"input": {"mode": "blocking"}, "output": {"mode": "blocking"}}},
    }
    cfg.update(overrides)
    return cfg


def test_valid_config():
    validate_config(_cfg())  # 不应抛异常


def test_placeholder_key_raises():
    with pytest.raises(ConfigError):
        validate_config(_cfg(llm={"api_key": "sk-your-api-key-here"}))


def test_missing_key_raises():
    with pytest.raises(ConfigError):
        validate_config(_cfg(llm={}))


def test_invalid_backend_raises():
    with pytest.raises(ConfigError):
        validate_config(_cfg(agent={"checkpoint": {"backend": "bogus"}}))


def test_invalid_mode_raises():
    with pytest.raises(ConfigError):
        validate_config(_cfg(harness={"guardrails": {"input": {"mode": "harsh"}}}))
