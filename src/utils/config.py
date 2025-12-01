"""
Configuration Management Module
================================

This module provides a robust configuration management system that:
    - Loads YAML configuration files with inheritance support
    - Validates configuration against schemas
    - Provides type-safe access to configuration values
    - Supports environment variable overrides
    - Enables runtime configuration updates

Example Usage:
    >>> from src.utils.config import load_config, Config
    >>> config = load_config("config/default.yaml")
    >>> print(config.preprocessing.face.backend)
    'mediapipe'
    >>> print(config.get("preprocessing.face.min_confidence", 0.5))
    0.7

Architecture:
    The Config class wraps a nested dictionary structure and provides
    attribute-style access. Configuration files can extend other files
    using the '_extends' key, enabling hierarchical configuration.
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar

import yaml

# Type variable for generic methods
T = TypeVar("T")


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class Config:
    """
    Type-safe configuration wrapper with attribute-style access.

    This class wraps a nested dictionary and provides convenient access
    patterns including attribute access, path-based access, and defaults.

    Attributes:
        _data: The underlying configuration dictionary.
        _path: The path to the configuration file (if loaded from file).

    Example:
        >>> config = Config({"model": {"name": "resnet50", "pretrained": True}})
        >>> config.model.name
        'resnet50'
        >>> config.get("model.layers", 50)
        50
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize configuration from dictionary.

        Args:
            data: Configuration dictionary. Defaults to empty dict.
            path: Path to the configuration file (for reference).
        """
        object.__setattr__(self, "_data", data or {})
        object.__setattr__(self, "_path", Path(path) if path else None)

    def __getattr__(self, name: str) -> Any:
        """
        Access configuration values as attributes.

        Args:
            name: The configuration key to access.

        Returns:
            The configuration value, wrapped in Config if it's a dict.

        Raises:
            AttributeError: If the key doesn't exist.
        """
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        try:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(
                f"Configuration has no attribute '{name}'. "
                f"Available keys: {list(self._data.keys())}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set configuration values as attributes.

        Args:
            name: The configuration key to set.
            value: The value to set.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in configuration."""
        return key in self._data

    def __iter__(self):
        """Iterate over configuration keys."""
        return iter(self._data)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._data})"

    def __len__(self) -> int:
        """Return number of top-level keys."""
        return len(self._data)

    def get(self, path: str, default: T = None) -> Union[Any, T]:
        """
        Get a configuration value using dot-notation path.

        This method allows accessing nested values using a dot-separated
        path string, with an optional default value.

        Args:
            path: Dot-separated path to the value (e.g., "model.encoder.dim").
            default: Default value if path doesn't exist.

        Returns:
            The configuration value or default.

        Example:
            >>> config.get("preprocessing.face.backend", "haar")
            'mediapipe'
            >>> config.get("nonexistent.path", "default")
            'default'
        """
        keys = path.split(".")
        value = self._data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value using dot-notation path.

        Creates intermediate dictionaries if they don't exist.

        Args:
            path: Dot-separated path to the value.
            value: The value to set.

        Example:
            >>> config.set("model.new_param", 100)
        """
        keys = path.split(".")
        data = self._data

        for key in keys[:-1]:
            if key not in data or not isinstance(data[key], dict):
                data[key] = {}
            data = data[key]

        data[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a plain dictionary.

        Returns:
            A deep copy of the configuration dictionary.
        """
        return copy.deepcopy(self._data)

    def keys(self) -> List[str]:
        """Return list of top-level configuration keys."""
        return list(self._data.keys())

    def items(self):
        """Return items iterator for top-level configuration."""
        return self._data.items()

    def update(self, other: Union[Dict[str, Any], "Config"]) -> None:
        """
        Update configuration with values from another dict or Config.

        Performs a deep merge, preserving nested structures.

        Args:
            other: Dictionary or Config to merge.
        """
        if isinstance(other, Config):
            other = other.to_dict()
        self._data = deep_merge(self._data, other)

    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against a schema.

        Args:
            schema: Validation schema dictionary.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        # Basic validation - check required fields
        for key, spec in schema.items():
            if isinstance(spec, dict):
                if spec.get("required", False) and key not in self._data:
                    errors.append(f"Required field '{key}' is missing")
                if key in self._data and "type" in spec:
                    expected_type = spec["type"]
                    actual_value = self._data[key]
                    if not isinstance(actual_value, expected_type):
                        errors.append(
                            f"Field '{key}' should be {expected_type.__name__}, "
                            f"got {type(actual_value).__name__}"
                        )
        return errors


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Values from 'override' take precedence. Nested dictionaries are
    recursively merged rather than replaced.

    Args:
        base: Base dictionary.
        override: Dictionary with override values.

    Returns:
        Merged dictionary.

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 4, "e": 5}}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 4, 'd': 3, 'e': 5}}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        ConfigError: If file cannot be read or parsed.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {path}: {e}")
    except IOError as e:
        raise ConfigError(f"Failed to read configuration file {path}: {e}")


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "DEEPFAKE_",
) -> Config:
    """
    Load configuration from YAML file with inheritance and overrides.

    This function supports:
        - YAML file loading
        - Configuration inheritance via '_extends' key
        - Environment variable overrides
        - Programmatic overrides

    Args:
        path: Path to the configuration file.
        overrides: Dictionary of values to override.
        env_prefix: Prefix for environment variable overrides.

    Returns:
        Loaded and merged Config object.

    Example:
        >>> config = load_config("config/default.yaml")
        >>> config = load_config(
        ...     "config/training.yaml",
        ...     overrides={"training.batch_size": 64}
        ... )
    """
    path = Path(path)
    data = load_yaml(path)

    # Handle configuration inheritance
    if "_extends" in data:
        extends_path = data.pop("_extends")
        # Resolve relative path from current config's directory
        if not Path(extends_path).is_absolute():
            extends_path = path.parent / extends_path
        base_data = load_yaml(extends_path)
        data = deep_merge(base_data, data)

    # Apply environment variable overrides
    data = apply_env_overrides(data, env_prefix)

    # Apply programmatic overrides
    if overrides:
        # Convert dot-notation overrides to nested dict
        nested_overrides = {}
        for key, value in overrides.items():
            if "." in key:
                keys = key.split(".")
                current = nested_overrides
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
            else:
                nested_overrides[key] = value
        data = deep_merge(data, nested_overrides)

    return Config(data, path)


def apply_env_overrides(
    data: Dict[str, Any], prefix: str, path: str = ""
) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables are expected in the format:
        {PREFIX}_{PATH} where PATH uses underscores for nesting.

    Example:
        DEEPFAKE_PREPROCESSING_FACE_BACKEND=haar

    Args:
        data: Configuration dictionary.
        prefix: Environment variable prefix.
        path: Current path in configuration (for recursion).

    Returns:
        Configuration with environment overrides applied.
    """
    result = copy.deepcopy(data)

    for key, value in result.items():
        current_path = f"{path}_{key}".upper() if path else key.upper()
        env_key = f"{prefix}{current_path}"

        if isinstance(value, dict):
            result[key] = apply_env_overrides(value, prefix, current_path)
        else:
            env_value = os.environ.get(env_key)
            if env_value is not None:
                # Parse the environment variable value
                result[key] = parse_env_value(env_value, type(value))

    return result


def parse_env_value(value: str, target_type: type) -> Any:
    """
    Parse an environment variable string to the target type.

    Args:
        value: String value from environment variable.
        target_type: Expected type of the configuration value.

    Returns:
        Parsed value in the correct type.
    """
    if target_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == list:
        # Parse comma-separated values
        return [v.strip() for v in value.split(",")]
    else:
        return value


def get_config_path(name: str = "default") -> Path:
    """
    Get the path to a named configuration file.

    Searches in standard locations:
        1. Current directory / config /
        2. Package directory / config /
        3. User config directory

    Args:
        name: Configuration name (without .yaml extension).

    Returns:
        Path to the configuration file.

    Raises:
        ConfigError: If configuration file not found.
    """
    # Possible config locations
    locations = [
        Path("config") / f"{name}.yaml",
        Path(__file__).parent.parent.parent / "config" / f"{name}.yaml",
        Path.home() / ".config" / "deepfake_detector" / f"{name}.yaml",
    ]

    for loc in locations:
        if loc.exists():
            return loc

    raise ConfigError(
        f"Configuration '{name}' not found. Searched locations: "
        f"{[str(p) for p in locations]}"
    )


# Default configuration singleton
_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """
    Get the default configuration singleton.

    Loads configuration on first access and caches it.

    Returns:
        The default Config object.
    """
    global _default_config
    if _default_config is None:
        try:
            _default_config = load_config(get_config_path("default"))
        except ConfigError:
            # Return empty config if no file found
            _default_config = Config({})
    return _default_config
