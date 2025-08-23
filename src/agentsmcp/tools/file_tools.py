"""File operation tools for OpenAI Agents SDK integration."""

from pathlib import Path
from typing import Any, Dict

from .base_tools import BaseTool, tool_registry


class FileOperationTool(BaseTool):
    """Base class for file operation tools."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and return Path object."""
        path = Path(file_path)
        if not path.is_absolute():
            # Make relative paths relative to current working directory
            path = Path.cwd() / path
        return path


class ReadFileTool(FileOperationTool):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read contents of a file. Returns the file content as a string.",
        )

    def execute(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        try:
            path = self._validate_file_path(file_path)

            if not path.exists():
                return f"Error: File does not exist: {file_path}"

            if not path.is_file():
                return f"Error: Path is not a file: {file_path}"

            with open(path, "r", encoding=encoding) as f:
                content = f.read()

            self.logger.debug(f"Read file: {file_path} ({len(content)} characters)")
            return content

        except PermissionError:
            return f"Error: Permission denied reading file: {file_path}"
        except UnicodeDecodeError:
            return f"Error: Unable to decode file with encoding {encoding}: {file_path}"
        except Exception as e:
            self.logger.exception(f"Error reading file {file_path}")
            return f"Error reading file {file_path}: {str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
            },
            "required": ["file_path"],
        }


class WriteFileTool(FileOperationTool):
    """Tool for writing file contents."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file. Creates directories if needed.",
        )

    def execute(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> str:
        """Write content to file."""
        try:
            path = self._validate_file_path(file_path)

            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            self.logger.debug(f"Wrote file: {file_path} ({len(content)} characters)")
            return f"Successfully wrote {len(content)} characters to {file_path}"

        except PermissionError:
            return f"Error: Permission denied writing to file: {file_path}"
        except Exception as e:
            self.logger.exception(f"Error writing file {file_path}")
            return f"Error writing file {file_path}: {str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Whether to create parent directories (default: true)",
                    "default": True,
                },
            },
            "required": ["file_path", "content"],
        }


class ListDirectoryTool(FileOperationTool):
    """Tool for listing directory contents."""

    def __init__(self):
        super().__init__(
            name="list_directory",
            description="List contents of a directory with file/directory information.",
        )

    def execute(self, directory_path: str, show_hidden: bool = False) -> str:
        """List directory contents."""
        try:
            path = self._validate_file_path(directory_path)

            if not path.exists():
                return f"Error: Directory does not exist: {directory_path}"

            if not path.is_dir():
                return f"Error: Path is not a directory: {directory_path}"

            items = []
            for item in path.iterdir():
                if not show_hidden and item.name.startswith("."):
                    continue

                item_type = "DIR" if item.is_dir() else "FILE"
                size = ""
                if item.is_file():
                    try:
                        size = f"({item.stat().st_size} bytes)"
                    except (OSError, AttributeError):
                        size = ""

                items.append(f"{item_type:4} {item.name} {size}")

            if not items:
                return f"Directory is empty: {directory_path}"

            result = f"Contents of {directory_path}:\n" + "\n".join(sorted(items))
            self.logger.debug(
                f"Listed directory: {directory_path} ({len(items)} items)"
            )
            return result

        except PermissionError:
            return f"Error: Permission denied accessing directory: {directory_path}"
        except Exception as e:
            self.logger.exception(f"Error listing directory {directory_path}")
            return f"Error listing directory {directory_path}: {str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Whether to show hidden files (default: false)",
                    "default": False,
                },
            },
            "required": ["directory_path"],
        }


# Create and register tool instances
read_file_tool = ReadFileTool()
write_file_tool = WriteFileTool()
list_directory_tool = ListDirectoryTool()

tool_registry.register(read_file_tool)
tool_registry.register(write_file_tool)
tool_registry.register(list_directory_tool)
