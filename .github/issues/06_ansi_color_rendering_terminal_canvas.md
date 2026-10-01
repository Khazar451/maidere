---
title: "[UX]: ANSI Color Sequence Rendering for Terminal Monospace Canvas"
labels: ["enhancement", "ui", "frontend"]
---

### Problem Statement
When background tools, test runners, or Uvicorn server logs emit standard ANSI color escape codes (e.g. `\x1b[32mPASS\x1b[0m`, `\x1b[31mERROR\x1b[0m`), `static/app.js` runs `escapeHTML(text)` and appends the line directly to the terminal canvas:

```javascript
// static/app.js: line 280
line.innerHTML = `<span class="term-ts">[${timeStr}]</span> <span class="term-text ${type}">${escapeHTML(text)}</span>`;
```

As a result, raw ANSI characters (such as `[32m` or `\u001b`) appear as visible clutter in the Agent Shell and Server Logs tabs rather than displaying syntax-highlighted colored text.

### Proposed Solution
1. **Lightweight ANSI-to-HTML Parser**:
   Add a pure-JavaScript utility in `static/app.js` to convert ANSI color codes into styled HTML spans:
   ```javascript
   function ansiToHtml(text) {
     const ansiMap = {
       30: 'ansi-black', 31: 'ansi-red', 32: 'ansi-green', 33: 'ansi-yellow',
       34: 'ansi-blue', 35: 'ansi-magenta', 36: 'ansi-cyan', 37: 'ansi-white',
       90: 'ansi-bright-black', 91: 'ansi-bright-red', 92: 'ansi-bright-green',
       // ...
     };
     // Replace ANSI sequences with corresponding span tags
   }
   ```
2. **Buffer Rolling Limit**:
   Cap DOM nodes inside `#terminal-canvas` to the most recent 1,000 lines, removing oldest lines to prevent browser memory leaks during extended background tasks.

3. **Terminal Palette Styling**:
   Define high-contrast ANSI colors in `static/style.css` consistent with Maidere's dark terminal theme.

### Acceptance Criteria
- [ ] Subprocess and server logs with ANSI colors render cleanly in terminal tabs without raw escape codes.
- [ ] Terminal buffer prunes older entries beyond 1,000 lines to ensure constant DOM performance.
