"""Tests for the terminal canvas ANSI renderer (static/ansi.js), run through Node."""

import json
import shutil
import subprocess
import unittest

from maidere.paths import static_dir

NODE = shutil.which("node")


def _run(expr: str, *args: str) -> str:
    script = f"const a = require({json.dumps(str(static_dir() / 'ansi.js'))}); process.stdout.write(String({expr}));"
    return subprocess.run([NODE, "-e", script, *args], capture_output=True, text=True, check=True, timeout=20).stdout


def render(text: str) -> str:
    return _run("a.ansiToHtml(JSON.parse(process.argv[1]))", json.dumps(text))


@unittest.skipUnless(NODE, "node is required to execute the browser-side renderer")
class TestAnsiRenderer(unittest.TestCase):
    def test_basic_colours_become_classes(self):
        self.assertEqual(render("\x1b[32mPASS\x1b[0m ok"), '<span class="ansi-green">PASS</span> ok')
        self.assertEqual(render("\x1b[91mfail\x1b[0m"), '<span class="ansi-bright-red">fail</span>')

    def test_no_raw_escape_artifacts_remain(self):
        out = render("\x1b[31mERROR\x1b[0m done\x1b[K\x1b[2J")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("[31m", out)
        self.assertNotIn("[0m", out)

    def test_plain_text_is_untouched_apart_from_escaping(self):
        self.assertEqual(render("a < b & c"), "a &lt; b &amp; c")

    def test_markup_in_logs_cannot_inject_html(self):
        out = render('\x1b[31m<img src=x onerror="alert(1)">\x1b[0m')
        self.assertNotIn("<img", out)
        self.assertIn("&lt;img", out)

    def test_truecolor_values_are_clamped_integers(self):
        out = render("\x1b[38;2;10;20;9999mx\x1b[0m")
        self.assertEqual(out, '<span style="color:rgb(10,20,255)">x</span>')

    def test_256_colour_palette(self):
        self.assertEqual(render("\x1b[38;5;196mr\x1b[0m"), '<span style="color:rgb(255,0,0)">r</span>')
        self.assertEqual(render("\x1b[38;5;2mg\x1b[0m"), '<span class="ansi-green">g</span>')

    def test_attributes_and_reset(self):
        self.assertEqual(
            render("\x1b[1;31mE\x1b[0mx"),
            '<span class="ansi-bold ansi-red">E</span>x',
        )
        self.assertEqual(render("\x1b[4mu\x1b[24mv"), '<span class="ansi-underline">u</span>v')

    def test_cursor_and_osc_sequences_are_dropped(self):
        self.assertEqual(render("\x1b[2K\x1b[1Gplain \x1b]0;title\x07text"), "plain text")

    def test_unterminated_escape_does_not_leak(self):
        self.assertNotIn("\x1b", render("tail\x1b"))
        self.assertEqual(render("tail\x1b[31"), "tail")

    def test_strip_ansi_for_copy(self):
        self.assertEqual(_run("a.stripAnsi(JSON.parse(process.argv[1]))", json.dumps("\x1b[31mred\x1b[0m!")), "red!")


class TestTerminalWiring(unittest.TestCase):
    """The page must load the renderer and cap the terminal buffer."""

    def test_assets_are_wired(self):
        index = (static_dir() / "index.html").read_text()
        self.assertLess(index.index("/static/ansi.js"), index.index("/static/app.js"))
        app = (static_dir() / "app.js").read_text()
        self.assertIn("MAX_TERMINAL_LINES = 1000", app)
        self.assertIn("MaidereAnsi", app)
        css = (static_dir() / "style.css").read_text()
        for cls in ("ansi-red", "ansi-green", "ansi-bright-red", "ansi-bg-red", "ansi-bold"):
            self.assertIn(f".{cls}", css)


if __name__ == "__main__":
    unittest.main()
