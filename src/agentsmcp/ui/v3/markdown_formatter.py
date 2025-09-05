"""Enhanced markdown formatter with color themes for plain text interface."""

import re
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ColorTheme:
    """Color theme for markdown rendering."""
    # Text styles
    heading1: str = "\033[1;36m"      # Bold cyan
    heading2: str = "\033[1;35m"      # Bold magenta  
    heading3: str = "\033[1;33m"      # Bold yellow
    heading4: str = "\033[1;32m"      # Bold green
    heading5: str = "\033[1;34m"      # Bold blue
    heading6: str = "\033[1;37m"      # Bold white
    
    # Emphasis
    bold: str = "\033[1m"             # Bold
    italic: str = "\033[3m"           # Italic
    strikethrough: str = "\033[9m"    # Strikethrough
    
    # Code and quotes
    code: str = "\033[48;5;236m\033[93m"  # Yellow on dark gray bg
    code_block: str = "\033[48;5;235m\033[92m"  # Green on darker bg
    blockquote: str = "\033[36m"      # Cyan
    
    # Lists and structure
    list_bullet: str = "\033[94m"     # Bright blue
    list_number: str = "\033[95m"     # Bright magenta
    horizontal_rule: str = "\033[90m" # Dark gray
    
    # Links and special
    link_url: str = "\033[94m\033[4m" # Blue underlined
    link_text: str = "\033[96m"       # Bright cyan
    
    # Progress and status
    progress_bar: str = "\033[42m"    # Green background
    progress_empty: str = "\033[100m" # Dark gray background
    success: str = "\033[92m"         # Bright green
    warning: str = "\033[93m"         # Bright yellow
    error: str = "\033[91m"           # Bright red
    info: str = "\033[94m"            # Bright blue
    
    # Reset
    reset: str = "\033[0m"            # Reset all formatting


class MarkdownFormatter:
    """Enhanced markdown formatter for terminal display with table support."""
    
    def __init__(self, theme: Optional[ColorTheme] = None, supports_color: bool = True):
        self.theme = theme or ColorTheme()
        self.supports_color = supports_color
        if not supports_color:
            # Disable all colors if not supported
            self.theme = ColorTheme(**{k: "" for k in ColorTheme.__dataclass_fields__})
    
    def format_text(self, text: str) -> str:
        """Format markdown text with color highlighting."""
        if not text:
            return ""
        
        lines = text.split('\n')
        formatted_lines = []
        
        in_code_block = False
        code_block_lang = ""
        in_table = False
        table_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_lang = line.strip()[3:].strip()
                    formatted_lines.append(f"{self.theme.code_block}┌─ {code_block_lang or 'text'} ─{'─' * (54 - len(code_block_lang or 'text'))}{self.theme.reset}")
                else:
                    in_code_block = False
                    code_block_lang = ""
                    formatted_lines.append(f"{self.theme.code_block}{'─' * 60}{self.theme.reset}")
                i += 1
                continue
            
            # Handle table detection and formatting
            if self._is_table_line(line) and not in_code_block:
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
                
                # Look ahead to see if table continues
                next_is_table = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    next_is_table = self._is_table_line(next_line) or self._is_table_separator(next_line)
                
                if not next_is_table:
                    # End of table - format it
                    formatted_table = self._format_table(table_lines)
                    formatted_lines.extend(formatted_table)
                    in_table = False
                    table_lines = []
                
                i += 1
                continue
            
            # Handle table separator lines
            if self._is_table_separator(line) and in_table:
                table_lines.append(line)
                i += 1
                continue
            
            # If we were in a table but this line isn't table-related, end the table
            if in_table:
                formatted_table = self._format_table(table_lines)
                formatted_lines.extend(formatted_table)
                in_table = False
                table_lines = []
            
            # Regular line formatting
            formatted_line = self._format_line(line, in_code_block, code_block_lang)
            formatted_lines.append(formatted_line)
            i += 1
        
        # Handle any remaining table at end of text
        if in_table and table_lines:
            formatted_table = self._format_table(table_lines)
            formatted_lines.extend(formatted_table)
        
        return '\n'.join(formatted_lines)
    
    def _is_table_line(self, line: str) -> bool:
        """Check if a line is part of a table."""
        stripped = line.strip()
        return bool(stripped and '|' in stripped and not stripped.startswith('|') == stripped.endswith('|'))
    
    def _is_table_separator(self, line: str) -> bool:
        """Check if a line is a table separator (like |---|---|)."""
        stripped = line.strip()
        if not stripped or '|' not in stripped:
            return False
        
        # Remove pipes and check if remaining parts are all dashes/colons/spaces
        parts = [part.strip() for part in stripped.split('|') if part.strip()]
        return all(re.match(r'^:?-+:?$', part) for part in parts)
    
    def _format_table(self, table_lines: List[str]) -> List[str]:
        """Format a markdown table with proper alignment and colors."""
        if not table_lines:
            return []
        
        # Parse table structure
        rows = []
        separator_index = -1
        
        for i, line in enumerate(table_lines):
            stripped = line.strip()
            if self._is_table_separator(stripped):
                separator_index = i
                continue
            
            # Split by | and clean up cells
            cells = [cell.strip() for cell in stripped.split('|')]
            # Remove empty cells from start/end if they exist
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            rows.append(cells)
        
        if not rows:
            return []
        
        # Calculate column widths
        max_cols = max(len(row) for row in rows)
        col_widths = []
        
        for col in range(max_cols):
            max_width = 0
            for row in rows:
                if col < len(row):
                    # Remove any existing color codes for width calculation
                    clean_text = re.sub(r'\033\[[0-9;]*m', '', row[col])
                    max_width = max(max_width, len(clean_text))
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        formatted_rows = []
        
        # Format each row
        for i, row in enumerate(rows):
            formatted_cells = []
            for j, cell in enumerate(row):
                if j < len(col_widths):
                    width = col_widths[j]
                    # Format cell content
                    formatted_cell = self._format_inline(cell) if cell else ""
                    
                    # Calculate padding (account for ANSI codes)
                    clean_cell = re.sub(r'\033\[[0-9;]*m', '', formatted_cell)
                    padding = width - len(clean_cell)
                    
                    if i == 0:  # Header row
                        formatted_cell = f"{self.theme.heading3}{formatted_cell}{self.theme.reset}"
                        # Recalculate padding after adding header formatting
                        clean_cell = re.sub(r'\033\[[0-9;]*m', '', formatted_cell)
                        padding = width - len(clean_cell) + len(f"{self.theme.heading3}{self.theme.reset}")
                    
                    formatted_cells.append(formatted_cell + " " * max(0, padding))
            
            # Build the row with borders
            row_text = f"{self.theme.list_bullet}│{self.theme.reset} " + f" {self.theme.list_bullet}│{self.theme.reset} ".join(formatted_cells) + f" {self.theme.list_bullet}│{self.theme.reset}"
            formatted_rows.append(row_text)
            
            # Add separator after header
            if i == 0 and len(rows) > 1:
                separator = f"{self.theme.list_bullet}├" + "─┼─".join("─" * w for w in col_widths) + f"┤{self.theme.reset}"
                formatted_rows.append(separator)
        
        # Add top and bottom borders
        top_border = f"{self.theme.list_bullet}┌" + "─┬─".join("─" * w for w in col_widths) + f"┐{self.theme.reset}"
        bottom_border = f"{self.theme.list_bullet}└" + "─┴─".join("─" * w for w in col_widths) + f"┘{self.theme.reset}"
        
        return [top_border] + formatted_rows + [bottom_border]
    
    def _format_line(self, line: str, in_code_block: bool, code_block_lang: str) -> str:
        """Format a single line of markdown."""
        if not line.strip():
            return line
        
        # Handle code blocks
        if in_code_block:
            return f"{self.theme.code_block}{line}{self.theme.reset}"
        
        # Handle other markdown elements
        formatted = line
        
        # Headers (must be done first to avoid conflicts)
        formatted = self._format_headers(formatted)
        
        # Horizontal rules
        if re.match(r'^[-*_]{3,}$', formatted.strip()):
            return f"{self.theme.horizontal_rule}{'─' * 60}{self.theme.reset}"
        
        # Lists (including numbered lists)
        formatted = self._format_lists(formatted)
        
        # Blockquotes
        formatted = self._format_blockquotes(formatted)
        
        # Inline formatting
        formatted = self._format_inline(formatted)
        
        return formatted
    
    def _format_headers(self, line: str) -> str:
        """Format markdown headers including numbered headers."""
        # Handle ATX headers (# ## ###)
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2)
            return self._format_header_text(text, level)
        
        # Handle Setext headers (underlined with = or -)
        # This would require looking ahead, so we'll handle it in the main loop if needed
        
        return line
    
    def _format_header_text(self, text: str, level: int) -> str:
        """Format header text with appropriate styling."""
        colors = [
            self.theme.heading1, self.theme.heading2, self.theme.heading3,
            self.theme.heading4, self.theme.heading5, self.theme.heading6
        ]
        color = colors[level - 1] if level <= 6 else self.theme.heading6
        
        # Extract any numbering at the start
        number_match = re.match(r'^(\d+(?:\.\d+)*\.?)\s*(.+)$', text)
        if number_match:
            number = number_match.group(1)
            title = number_match.group(2)
            formatted_number = f"{self.theme.list_number}{number}{self.theme.reset}"
            formatted_text = f"{color}{title}{self.theme.reset}"
            text = f"{formatted_number} {formatted_text}"
        else:
            text = f"{color}{text}{self.theme.reset}"
        
        # Add decorative elements based on level
        if level == 1:
            decoration = "═" * min(len(re.sub(r'\033\[[0-9;]*m', '', text)), 50)
            return f"{text}\n{color}{decoration}{self.theme.reset}"
        elif level == 2:
            decoration = "─" * min(len(re.sub(r'\033\[[0-9;]*m', '', text)), 50)
            return f"{text}\n{color}{decoration}{self.theme.reset}"
        elif level == 3:
            return f"▶ {text}"
        else:
            return f"{'  ' * (level - 3)}• {text}"
    
    def _format_lists(self, line: str) -> str:
        """Format markdown lists including nested and numbered lists."""
        # Handle ordered lists with better number formatting
        match = re.match(r'^(\s*)(\d+(?:\.\d+)*\.?)\s+(.+)$', line)
        if match:
            indent = match.group(1)
            number = match.group(2)
            content = match.group(3)
            
            # Format the number with color
            formatted_number = f"{self.theme.list_number}{number}{self.theme.reset}"
            formatted_content = self._format_inline(content)
            
            return f"{indent}{formatted_number} {formatted_content}"
        
        # Handle unordered lists
        match = re.match(r'^(\s*)[-*+]\s+(.+)$', line)
        if match:
            indent = match.group(1)
            content = match.group(2)
            
            # Choose bullet style based on indent level
            indent_level = len(indent) // 2
            bullets = ["•", "◦", "▪", "▫"]
            bullet = bullets[min(indent_level, len(bullets) - 1)]
            
            formatted_bullet = f"{self.theme.list_bullet}{bullet}{self.theme.reset}"
            formatted_content = self._format_inline(content)
            
            return f"{indent}{formatted_bullet} {formatted_content}"
        
        return line
    
    def _format_blockquotes(self, line: str) -> str:
        """Format markdown blockquotes."""
        match = re.match(r'^(>\s*)(.*)$', line)
        if match:
            prefix = match.group(1)
            content = match.group(2)
            formatted_content = self._format_inline(content)
            return f"{self.theme.blockquote}▌ {formatted_content}{self.theme.reset}"
        
        return line
    
    def _format_inline(self, line: str) -> str:
        """Format inline markdown elements with improved bold handling."""
        # Handle bold with ** (higher priority)
        line = re.sub(
            r'\*\*([^*]+)\*\*',
            f'{self.theme.bold}\\1{self.theme.reset}',
            line
        )
        
        # Handle bold with __ (alternative syntax)
        line = re.sub(
            r'__([^_]+)__',
            f'{self.theme.bold}\\1{self.theme.reset}',
            line
        )
        
        # Handle italic with * (avoiding conflict with bold)
        line = re.sub(
            r'(?<!\*)\*([^*\s][^*]*[^*\s])\*(?!\*)',
            f'{self.theme.italic}\\1{self.theme.reset}',
            line
        )
        
        # Handle italic with _ (alternative syntax)
        line = re.sub(
            r'(?<!_)_([^_\s][^_]*[^_\s])_(?!_)',
            f'{self.theme.italic}\\1{self.theme.reset}',
            line
        )
        
        # Inline code (backticks)
        line = re.sub(
            r'`([^`]+)`',
            f'{self.theme.code} \\1 {self.theme.reset}',
            line
        )
        
        # Links [text](url)
        line = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            f'{self.theme.link_text}\\1{self.theme.reset} {self.theme.link_url}(\\2){self.theme.reset}',
            line
        )
        
        # Strikethrough
        line = re.sub(
            r'~~([^~]+)~~',
            f'{self.theme.strikethrough}\\1{self.theme.reset}',
            line
        )
        
        return line
    
    def format_progress_bar(self, current: int, total: int, width: int = 30, label: str = "") -> str:
        """Create a colored progress bar."""
        if total <= 0:
            return f"{label} [{'?' * width}]"
        
        filled = int((current / total) * width)
        empty = width - filled
        
        bar = (f"{self.theme.progress_bar}{' ' * filled}{self.theme.reset}"
               f"{self.theme.progress_empty}{' ' * empty}{self.theme.reset}")
        
        percentage = int((current / total) * 100)
        status = f"{current}/{total} ({percentage}%)"
        
        return f"{label} [{bar}] {status}"
    
    def format_status_message(self, message: str, status: str = "info") -> str:
        """Format a status message with appropriate coloring."""
        colors = {
            "success": self.theme.success,
            "warning": self.theme.warning,
            "error": self.theme.error,
            "info": self.theme.info
        }
        
        color = colors.get(status, self.theme.info)
        icons = {
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "info": "ℹ️ "
        }
        
        icon = icons.get(status, "ℹ️ ")
        return f"{color}{icon} {message}{self.theme.reset}"
    
    def format_thinking_step(self, step_num: int, total_steps: int, thought: str) -> str:
        """Format a sequential thinking step."""
        progress = self.format_progress_bar(step_num, total_steps, 20, "Thinking")
        header = f"{self.theme.heading3}🧠 Step {step_num}/{total_steps}{self.theme.reset}"
        
        # Format the thought content as markdown
        formatted_thought = self.format_text(thought)
        
        return f"{header}\n{progress}\n{formatted_thought}\n"
    
    def format_agent_activity(self, agent_name: str, activity: str, progress: Optional[float] = None) -> str:
        """Format agent activity with optional progress."""
        header = f"{self.theme.heading4}🤖 {agent_name}{self.theme.reset}"
        
        if progress is not None:
            progress_bar = self.format_progress_bar(
                int(progress * 100), 100, 25, "Progress"
            )
            return f"{header}\n{activity}\n{progress_bar}"
        else:
            return f"{header}\n{activity}"