import pytest
import yaml
from pathlib import Path
from agent import load_yaml, validate_config, ValidationError

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def test_default_config_validates():
    config = load_yaml(CONFIG_PATH)
    assert isinstance(config, dict)
    assert validate_config(config) is True

def test_missing_system_prompt_fails(tmp_path):
    config = load_yaml(CONFIG_PATH)
    config.pop("system_prompt", None)
    # write a temp file and try to validate
    tmp = tmp_path / "cfg.yaml"
    tmp.write_text(yaml.safe_dump(config))
    with pytest.raises(ValidationError):
        validate_config(config)
