# AgentsMCP Design System

## Color Palette & Semantic Tokens

### Core Colors
```json
{
  "colors": {
    "brand": {
      "primary": "#6366f1",     // Indigo - main brand
      "secondary": "#8b5cf6",   // Purple - accent
      "tertiary": "#06b6d4"     // Cyan - highlight
    },
    "semantic": {
      "success": "#10b981",     // Green
      "warning": "#f59e0b",     // Amber
      "error": "#ef4444",       // Red
      "info": "#3b82f6"         // Blue
    },
    "neutral": {
      "50": "#fafafa",
      "100": "#f5f5f5",
      "200": "#e5e5e5",
      "300": "#d4d4d4",
      "400": "#a3a3a3",
      "500": "#737373",
      "600": "#525252",
      "700": "#404040",
      "800": "#262626",
      "900": "#171717",
      "950": "#0a0a0a"
    }
  }
}
```

### Terminal-Specific Colors
```json
{
  "terminal": {
    "background": {
      "dark": "#0a0a0a",        // neutral-950
      "light": "#fafafa"        // neutral-50
    },
    "foreground": {
      "dark": "#f5f5f5",        // neutral-100
      "light": "#171717"        // neutral-900
    },
    "accent": {
      "bright_blue": "#60a5fa", // For links and highlights
      "bright_green": "#34d399", // For success states
      "bright_yellow": "#fbbf24", // For warnings
      "bright_red": "#f87171",   // For errors
      "bright_purple": "#a78bfa", // For special states
      "bright_cyan": "#22d3ee"   // For info
    }
  }
}
```

## Typography Scale

### Font Stack
```css
:root {
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', Consolas, monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
```

### Type Scale (rem units)
```json
{
  "fontSize": {
    "xs": "0.75rem",    // 12px - Small labels
    "sm": "0.875rem",   // 14px - Body small
    "base": "1rem",     // 16px - Body text
    "lg": "1.125rem",   // 18px - Large body
    "xl": "1.25rem",    // 20px - H3
    "2xl": "1.5rem",    // 24px - H2
    "3xl": "1.875rem",  // 30px - H1
    "4xl": "2.25rem"    // 36px - Display
  },
  "lineHeight": {
    "tight": 1.25,
    "normal": 1.5,
    "relaxed": 1.75
  }
}
```

## Spacing System (4px base unit)

```json
{
  "spacing": {
    "0": "0",
    "1": "0.25rem",   // 4px
    "2": "0.5rem",    // 8px
    "3": "0.75rem",   // 12px
    "4": "1rem",      // 16px
    "5": "1.25rem",   // 20px
    "6": "1.5rem",    // 24px
    "8": "2rem",      // 32px
    "10": "2.5rem",   // 40px
    "12": "3rem",     // 48px
    "16": "4rem",     // 64px
    "20": "5rem",     // 80px
    "24": "6rem"      // 96px
  }
}
```

## Component Specifications

### Status Indicators
```
Agent Status Colors:
🟢 Active/Healthy    - #10b981 (success)
🟡 Warning/Degraded  - #f59e0b (warning)  
🔴 Error/Failed      - #ef4444 (error)
⚪ Idle/Inactive     - #737373 (neutral-500)
🔵 Processing        - #3b82f6 (info)
🟣 Maintenance       - #8b5cf6 (secondary)

Harmony Score Visual:
████████████████████████████████▒▒▒▒▒  (87%)
└─ Filled: #6366f1 (primary)
└─ Empty: #d4d4d4 (neutral-300)
```

### Progress Bars
```
Loading States:
[▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░] 67%

Animation: Left-to-right fill with subtle pulse
Colors: Primary (#6366f1) fill, neutral background
Height: 4px for small, 8px for prominent
```

### Buttons & Interactive Elements
```
Primary Button:
┌──────────────┐
│   Execute    │  ← Background: #6366f1, Text: white
└──────────────┘    Hover: #4f46e5, Active: #4338ca

Secondary Button:
┌──────────────┐
│     Edit     │  ← Border: #6366f1, Text: #6366f1, Background: transparent
└──────────────┘    Hover: Background: #6366f1/10

Danger Button:
┌──────────────┐
│   Restart    │  ← Background: #ef4444, Text: white
└──────────────┘    Hover: #dc2626, Active: #b91c1c
```

### Cards & Containers
```
┌─ Card Header ─────────────────────┐
│                                   │
│  Content area with proper         │
│  padding and visual hierarchy     │
│                                   │
└───────────────────────────────────┘

Border: 1px solid #e5e5e5 (neutral-200)
Border Radius: 8px
Padding: 24px (space-6)
Background: white (light) / #171717 (dark)
Shadow: 0 1px 3px rgba(0,0,0,0.1)
```

### Form Elements
```
Input Field:
┌─────────────────────────────────────┐
│ Enter command or natural language   │
└─────────────────────────────────────┘

Height: 48px (space-12)
Padding: 12px 16px (space-3 space-4)
Border: 1px solid #d4d4d4, focus: #6366f1
Border Radius: 6px
Background: white/dark with 2px focus ring
```

## Layout Grid & Breakpoints

### Terminal Viewport Sizes
```
Small Terminal:   80 columns × 24 rows   (minimum)
Medium Terminal: 120 columns × 40 rows   (standard)
Large Terminal:  160 columns × 60 rows   (wide)
XL Terminal:     200+ columns × 80+ rows (ultra-wide)
```

### Responsive Layout Rules
1. **Mobile-first**: Design for 80-column terminals
2. **Progressive enhancement**: Add features for larger viewports
3. **Graceful degradation**: Essential features work everywhere
4. **Content priority**: Most important info visible first

## Motion & Animation

### Timing Functions
```css
:root {
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
}
```

### Duration Scale
```json
{
  "duration": {
    "fast": "150ms",      // Quick feedback
    "normal": "250ms",    // Standard transitions
    "slow": "350ms",      // Complex animations
    "extra-slow": "500ms" // Page transitions
  }
}
```

### Animation Patterns
```
Loading Spinner: Smooth rotation, 1s duration
Progress Bars: Linear fill, eased transitions
State Changes: Opacity + transform, 250ms
Modal Entry: Scale up from 0.95 to 1.0, 200ms
Toast Notifications: Slide in from top, 300ms
```

## Accessibility Standards

### Contrast Ratios (WCAG 2.2 AA)
- **Normal text**: 4.5:1 minimum
- **Large text**: 3:1 minimum  
- **Interactive elements**: 3:1 minimum
- **Focus indicators**: 3:1 minimum against background

### Focus Management
```
Focus Ring: 2px solid #6366f1 with 2px offset
Keyboard Navigation: Tab order follows visual hierarchy
Skip Links: Available for screen readers
Focus Trapping: Modal dialogs trap focus
```

### Screen Reader Support
- Semantic HTML structure
- ARIA labels for all interactive elements
- Live regions for dynamic content updates
- Descriptive error messages
- Progress announcements

## Icon System

### Status Icons
```
🟢 ●  Success / Active      (U+1F7E2)
🟡 ●  Warning / Degraded    (U+1F7E1)  
🔴 ●  Error / Failed        (U+1F534)
⚪ ○  Idle / Inactive       (U+26AA)
🔵 ●  Processing / Info     (U+1F535)
🟣 ●  Special / Maintenance (U+1F7E3)
```

### Action Icons
```
▶️  Play / Execute         (U+25B6)
⏸️  Pause                  (U+23F8)
⏹️  Stop                   (U+23F9)
🔄  Refresh / Restart      (U+1F504)
⚙️  Settings / Configure   (U+2699)
📊  Analytics / Reports    (U+1F4CA)
🎵  Harmony / Music        (U+1F3B5)
🎭  Agent / Theater        (U+1F3AD)
```

### Sizing Guidelines
- Small: 16px (1rem)
- Medium: 24px (1.5rem)
- Large: 32px (2rem)
- Always maintain aspect ratio
- Ensure visibility at all sizes