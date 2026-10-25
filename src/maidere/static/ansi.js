/* ANSI SGR -> safe HTML for the terminal canvas.
 *
 * Text is HTML-escaped before any markup is added. The only inline style ever
 * emitted is a colour built from clamped integers, so log content cannot inject
 * attributes or tags. Non-colour escape sequences (cursor movement, OSC titles)
 * are dropped.
 */
(function (root) {
  'use strict';

  const BASIC = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'];
  // CSI ... final byte (any), OSC ... BEL/ST, lone two-byte escapes, and a CSI
  // cut off at the end of a chunk (so its parameters never show as text).
  const SEQUENCE = /\x1b(?:\[([0-9;:]*)([@-~])|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_]|\[[0-9;:]*$)/g;

  function escapeHTML(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function clamp255(n) {
    return Math.max(0, Math.min(255, n | 0));
  }

  // xterm 256-colour palette index -> [r, g, b] for indices 16..255.
  function palette256(n) {
    if (n >= 232) {
      const v = 8 + (n - 232) * 10;
      return [v, v, v];
    }
    const i = n - 16;
    const level = (x) => (x === 0 ? 0 : 55 + x * 40);
    return [level(Math.floor(i / 36)), level(Math.floor(i / 6) % 6), level(i % 6)];
  }

  function freshState() {
    return { bold: false, dim: false, italic: false, underline: false, fg: null, bg: null };
  }

  // A colour is {cls} for the 16 themed colours or {rgb:[r,g,b]} for the rest.
  function basicColor(index, bright, prefix) {
    return { cls: `ansi-${prefix}${bright ? 'bright-' : ''}${BASIC[index]}` };
  }

  function extendedColor(params, i) {
    // params[i] is 38/48; returns [color|null, consumed]
    const mode = params[i + 1];
    if (mode === 5 && i + 2 < params.length) {
      const n = clamp255(params[i + 2]);
      if (n < 16) return [{ index: n }, 3];
      return [{ rgb: palette256(n) }, 3];
    }
    if (mode === 2 && i + 4 < params.length) {
      return [{ rgb: [clamp255(params[i + 2]), clamp255(params[i + 3]), clamp255(params[i + 4])] }, 5];
    }
    return [null, 1];
  }

  function applyParams(state, rawParams) {
    const params = rawParams === '' ? [0] : rawParams.split(/[;:]/).map((p) => (p === '' ? 0 : parseInt(p, 10) || 0));
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      if (p === 0) Object.assign(state, freshState());
      else if (p === 1) state.bold = true;
      else if (p === 2) state.dim = true;
      else if (p === 3) state.italic = true;
      else if (p === 4) state.underline = true;
      else if (p === 22) state.bold = state.dim = false;
      else if (p === 23) state.italic = false;
      else if (p === 24) state.underline = false;
      else if (p >= 30 && p <= 37) state.fg = basicColor(p - 30, false, '');
      else if (p >= 90 && p <= 97) state.fg = basicColor(p - 90, true, '');
      else if (p >= 40 && p <= 47) state.bg = basicColor(p - 40, false, 'bg-');
      else if (p >= 100 && p <= 107) state.bg = basicColor(p - 100, true, 'bg-');
      else if (p === 39) state.fg = null;
      else if (p === 49) state.bg = null;
      else if (p === 38 || p === 48) {
        const [color, used] = extendedColor(params, i);
        if (color) {
          const target = p === 38 ? 'fg' : 'bg';
          const prefix = p === 38 ? '' : 'bg-';
          state[target] = color.index !== undefined ? basicColor(color.index % 8, color.index >= 8, prefix) : color;
        }
        i += used - 1;
      }
    }
  }

  function openTag(state) {
    const classes = [];
    const styles = [];
    if (state.bold) classes.push('ansi-bold');
    if (state.dim) classes.push('ansi-dim');
    if (state.italic) classes.push('ansi-italic');
    if (state.underline) classes.push('ansi-underline');
    if (state.fg) {
      if (state.fg.cls) classes.push(state.fg.cls);
      else styles.push(`color:rgb(${state.fg.rgb.join(',')})`);
    }
    if (state.bg) {
      if (state.bg.cls) classes.push(state.bg.cls);
      else styles.push(`background-color:rgb(${state.bg.rgb.join(',')})`);
    }
    if (!classes.length && !styles.length) return '';
    const cls = classes.length ? ` class="${classes.join(' ')}"` : '';
    const style = styles.length ? ` style="${styles.join(';')}"` : '';
    return `<span${cls}${style}>`;
  }

  function ansiToHtml(text) {
    if (typeof text !== 'string' || text === '') return '';
    const state = freshState();
    let out = '';
    let open = false;
    let last = 0;

    const emit = (chunk) => {
      if (!chunk) return;
      const tag = open ? '' : openTag(state);
      if (tag) {
        out += tag;
        open = true;
      }
      out += escapeHTML(chunk);
    };
    const close = () => {
      if (open) {
        out += '</span>';
        open = false;
      }
    };

    SEQUENCE.lastIndex = 0;
    let m;
    while ((m = SEQUENCE.exec(text)) !== null) {
      emit(text.slice(last, m.index));
      last = SEQUENCE.lastIndex;
      if (m[2] === 'm') {
        close();
        applyParams(state, m[1]);
      }
    }
    emit(text.slice(last));
    close();
    // Stray ESC bytes that did not form a sequence must not reach the DOM.
    return out.replace(/\x1b/g, '');
  }

  function stripAnsi(text) {
    if (typeof text !== 'string') return '';
    return text.replace(SEQUENCE, '').replace(/\x1b/g, '');
  }

  const api = { ansiToHtml, stripAnsi };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MaidereAnsi = api;
})(typeof window !== 'undefined' ? window : globalThis);
