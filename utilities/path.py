import logging
from pathlib import Path
from typing import Union
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_path(path_input: Union[str, Path, None]) -> Optional[Path]:
    """
    Ensure input is converted to Path object with proper error handling.

    Args:
        path_input: String path, Path object, or None

    Returns:
        Path object or None if input was None

    Raises:
        ValueError: If path_input is not a valid path type
    """
    if path_input is None:
        return None

    if isinstance(path_input, Path):
        return path_input

    if isinstance(path_input, str):
        if not path_input.strip():
            return None
        return Path(path_input)

    raise ValueError(
        f"Invalid path type: {type(path_input)}. Expected str, Path, or None."
    )


def ensure_path_exists(
    path_input: Union[str, Path], create_parents: bool = True
) -> Path:
    """
    Ensure path exists and return as Path object.

    Args:
        path_input: String path or Path object
        create_parents: Whether to create parent directories

    Returns:
        Path object

    Raises:
        ValueError: If path cannot be created or is invalid
    """
    path_obj = ensure_path(path_input)

    if path_obj is None:
        raise ValueError("Path cannot be None")

    try:
        if create_parents:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        return path_obj
    except Exception as e:
        raise ValueError(f"Failed to ensure path exists '{path_obj}': {str(e)}")


def safe_path_operation(func):
    """
    Decorator to safely handle path operations by converting string arguments to Path objects.
    """

    def wrapper(*args, **kwargs):
        # Convert string paths to Path objects in args
        new_args = []
        for arg in args:
            if isinstance(arg, str) and (
                "/" in arg
                or "\\" in arg
                or arg.endswith(
                    (".txt", ".json", ".jsonl", ".npz", ".md", ".pdf", ".docx")
                )
            ):
                new_args.append(Path(arg))
            else:
                new_args.append(arg)

        # Convert string paths to Path objects in kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if key.endswith(("_path", "_dir", "_file")) or key in (
                "path",
                "file_path",
                "dir_path",
                "output_path",
                "input_path",
            ):
                if isinstance(value, str):
                    new_kwargs[key] = Path(value)
                else:
                    new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


class PathHandler:
    """
    Context manager for safe path operations.
    """

    def __init__(self, *paths: Union[str, Path]):
        self.paths = [ensure_path(p) for p in paths]
        self.created_paths = []

    def __enter__(self):
        return self.paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup logic if needed
        pass

    def ensure_all_exist(self, create_parents: bool = True):
        """Ensure all paths exist."""
        for path in self.paths:
            if path is not None:
                try:
                    if create_parents:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self.created_paths.append(path.parent)
                except Exception as e:
                    logger.error(f"Failed to create path {path}: {str(e)}")
                    raise
