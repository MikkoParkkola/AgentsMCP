# AgentsMCP Accessibility Guidelines

## Universal Design Principles

### 1. Perceivable Information
All users must be able to perceive the information being presented.

#### Visual Design
- **High Contrast**: Minimum 4.5:1 ratio for normal text, 3:1 for large text
- **Color Independence**: Never rely solely on color to convey information
- **Scalable Text**: Support terminal font scaling up to 200% without loss of functionality
- **Visual Hierarchy**: Use consistent heading levels, spacing, and typography

#### Color-Blind Accessible Patterns
```
Instead of: "Red means error, Green means success"
Use: "🔴 Error: Connection failed" and "🟢 Success: Agent started"

Status Pattern:
🔴 ❌ Failed    - Icon + color + text
🟡 ⚠️  Warning  - Icon + color + text  
🟢 ✅ Success   - Icon + color + text
⚪ ⏸️  Paused   - Icon + color + text
```

### 2. Operable Interface
All interface components must be operable by all users.

#### Keyboard Navigation
```
Navigation Pattern:
Tab       - Next focusable element
Shift+Tab - Previous focusable element
Enter     - Activate button/link
Space     - Activate button, scroll page
Esc       - Cancel/close modal/menu
Arrow Keys- Navigate lists/grids/menus

Command Mode:
Ctrl+K    - Quick command palette
Ctrl+/    - Show all shortcuts
F1        - Context help
F2        - Rename/edit mode
F5        - Refresh/reload
```

#### Focus Management
```
Focus Indicators:
┌─────────────────┐
│  [Execute] ←────┤ 2px solid #6366f1 ring
└─────────────────┘

Focus Order:
1. Main navigation
2. Primary content
3. Secondary actions  
4. Footer/help

Modal Focus Trapping:
- Focus moves to first element in modal
- Tab cycles within modal only
- Esc returns focus to trigger element
```

### 3. Understandable Content
Information and UI operation must be understandable.

#### Clear Language
- **Plain Language**: Avoid jargon, use common terms
- **Consistent Terminology**: Same concept = same words throughout
- **Error Messages**: Specific, actionable guidance
- **Progressive Disclosure**: Show complexity gradually

#### Error Handling
```
Bad:  "Error: ECONNREFUSED"
Good: "Connection Error: Cannot reach the server at localhost:3000. 
       Try: 1) Check if the server is running, 2) Verify the port number"

Bad:  "Invalid input"
Good: "Command not recognized. Did you mean 'agentsmcp create'? 
       Type 'help' to see all available commands."
```

### 4. Robust Implementation
Content must be robust enough to work with assistive technologies.

#### Screen Reader Support

##### Semantic Structure
```
<h1>AgentsMCP - Multi-Agent Orchestra</h1>
<main aria-label="Agent Dashboard">
  <section aria-labelledby="status-heading">
    <h2 id="status-heading">Agent Status Overview</h2>
    <ul role="list" aria-label="Agent status list">
      <li role="listitem">
        <span aria-label="Python agent: healthy, 92% utilization">
          🟢 Python (92%)
        </span>
      </li>
    </ul>
  </section>
</main>
```

##### Live Regions for Dynamic Updates
```
<div aria-live="polite" aria-label="Status updates">
  <!-- Announced: "Agent Python-SEC has completed security scan" -->
</div>

<div aria-live="assertive" aria-label="Error notifications">
  <!-- Announced immediately: "Connection to Node-UI agent lost" -->
</div>

Priority Levels:
- aria-live="polite"     - Announce when user is idle
- aria-live="assertive"  - Announce immediately (errors)
- aria-live="off"        - Don't announce (default)
```

## Terminal-Specific Accessibility

### Screen Reader Compatibility

#### Content Structure
```
CLI Output Format:
=== AgentsMCP Status ===
Heading level 1: Agent Orchestra Status
- List item: Python agent, status healthy, utilization 92%
- List item: Node agent, status warning, high memory usage
- List item: Go agent, status error, connection timeout

Progress indication:
Task "Security Scan" progress: 67% complete, estimated 2 minutes remaining
```

#### Command Feedback
```
Command Echo:
User types: "create python agent"
Screen reader hears: 
1. "create python agent" (as user types)
2. "Command recognized: Create Python agent with security tools"
3. "Executing... Agent creation in progress"
4. "Success: Python agent 'security-checker' created and ready"
```

### Reduced Motion Support

#### Animation Controls
```css
@media (prefers-reduced-motion: reduce) {
  /* Disable spinning animations */
  .loading-spinner { animation: none; }
  
  /* Use fade instead of slide */
  .slide-animation { 
    transform: none;
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  
  /* Reduce progress bar animation speed */
  .progress-bar { animation-duration: 0.1s; }
}
```

#### Alternative Feedback
```
Motion-sensitive users get:
- Static progress indicators instead of animated bars
- Immediate state changes instead of transitions
- Text-based loading indicators instead of spinners
- Solid colors instead of pulsing/blinking effects
```

## Internationalization (i18n) Readiness

### Text Expansion Considerations
```
English: "Create Agent"      (12 characters)
German:  "Agent Erstellen"   (15 characters)
Spanish: "Crear Agente"      (12 characters)  
Russian: "Создать агента"    (14 characters)

Design for 50% text expansion in buttons and labels
Use flexible layouts that adapt to content length
```

### RTL (Right-to-Left) Support
```
LTR Layout:
┌─ Agents ────── Status ────── Actions ──┐
│  Python       🟢 Active    [Restart]   │
│  Node         🔴 Error     [Debug]     │
└─────────────────────────────────────────┘

RTL Layout:
┌── Actions ────── Status ────── Agents ─┐
│  [Restart]    Active 🟢       Python  │
│  [Debug]      Error 🔴        Node    │
└─────────────────────────────────────────┘
```

## Assistive Technology Testing

### Screen Reader Testing Checklist
- [ ] NVDA (Windows) - Free, widely used
- [ ] JAWS (Windows) - Industry standard
- [ ] VoiceOver (macOS) - Built-in, well-integrated
- [ ] Orca (Linux) - Open source option

### Testing Scenarios
1. **Navigation**: Can users find and access all features?
2. **Status Understanding**: Are agent states clearly communicated?
3. **Error Recovery**: Are error messages helpful and actionable?
4. **Command Input**: Is the natural language interface accessible?
5. **Progress Feedback**: Are long operations properly announced?

### Keyboard Testing Checklist
- [ ] All functionality accessible via keyboard
- [ ] Focus indicators visible and clear
- [ ] Logical tab order throughout interface
- [ ] No keyboard traps (except intentional focus management)
- [ ] Shortcut keys don't conflict with assistive technology

## Testing Tools & Validation

### Automated Testing
```bash
# Color contrast validation
npm install -g colour-contrast-checker
color-contrast-checker --foreground="#171717" --background="#fafafa"

# ARIA validation
npm install -g axe-cli  
axe https://localhost:3000 --tags wcag2a,wcag2aa

# Screen reader testing
# Use NVDA, JAWS, VoiceOver, or Orca for manual testing
```

### Manual Testing Protocol

#### Daily Accessibility Checks
1. **Keyboard Navigation**: Tab through entire interface
2. **Color Contrast**: Verify all text meets WCAG standards
3. **Screen Reader**: Test with one screen reader weekly
4. **Focus Management**: Check focus indicators and order

#### Pre-release Accessibility Audit
1. **Automated Scan**: Run axe-core or similar tool
2. **Expert Review**: Accessibility professional evaluation
3. **User Testing**: Test with real assistive technology users
4. **Documentation**: Update accessibility features documentation

## Emergency Accessibility Fixes

### Quick Fixes for Common Issues
```css
/* Missing focus indicators */
button:focus { outline: 2px solid #6366f1; }

/* Low contrast text */
.text-gray-500 { color: #525252; } /* Bumped from #737373 */

/* Missing alt text */
<img src="status.png" alt="Agent status: 3 active, 1 error">

/* Poor heading structure */
<h2>Status</h2> <!-- Changed from <div class="heading"> -->
```

### Accessibility Debt Management
- **Critical**: Blocks assistive technology users completely
- **High**: Creates significant barriers or confusion
- **Medium**: Causes minor inconvenience or extra steps
- **Low**: Minor improvements that enhance experience

Track accessibility issues in the same system as other bugs, with clear priority guidelines and regular audits.