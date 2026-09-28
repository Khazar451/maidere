"""Headless browser tool with Playwright zombie process prevention.

Pattern #4 Guarantees:
- Every Playwright invocation uses async with + try/finally
- No persistent module-level browser or page objects
- 15-second page load timeout
- Text content capped at 4000 characters (~1000 tokens for LLM context)
- Always closes browser instance even on crash or timeout
"""

from typing import Any
import structlog
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

from tools.base import BaseTool

import re

logger = structlog.get_logger()

MAX_CONTENT_CHARS: int = 2500
DEFAULT_TIMEOUT_MS: int = 15000


async def browse_url(url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    """Browse a URL and return sanitized page text. Guaranteed cleanup."""
    if not url or not url.strip():
        return "Error: Empty URL provided."

    target_url = url.strip()
    if not target_url.startswith(("http://", "https://")):
        target_url = f"https://{target_url}"

    if async_playwright is None:
        return "Error: Playwright is not installed. Install with 'pip install playwright' to enable browser scraping."

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(target_url, timeout=timeout_ms, wait_until="domcontentloaded")
                
                # Remove boilerplate navigation, header, footer, and script elements
                try:
                    await page.evaluate("""() => {
                        const junk = document.querySelectorAll('nav, header, footer, script, style, noscript, svg, [role="navigation"], [role="banner"], [role="contentinfo"]');
                        junk.forEach(el => el.remove());
                    }""")
                except Exception:
                    pass

                # Select main article container or fall back to body
                main_el = await page.query_selector("main, article, [role='main'], #content, .content, .main-content, .markdown")
                if main_el:
                    raw_content = await main_el.inner_text()
                else:
                    raw_content = await page.inner_text("body")
                
                # Sanitize whitespace, newlines, and repetitive formatting
                cleaned = re.sub(r"[ \t]+", " ", raw_content)
                cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned).strip()
                trimmed = cleaned[:MAX_CONTENT_CHARS]
                
                return trimmed if trimmed else "Web page loaded but contained no readable body text."

            finally:
                await browser.close()  # ALWAYS runs, even on timeout/crash
    except Exception as e:
        await logger.aerror("browser_failed", url=target_url, error=str(e))
        return f"Error browsing '{url}': {str(e)}"



class BrowserTool(BaseTool):
    """Tool to browse web pages using headless Playwright with guaranteed cleanup."""

    name: str = "browser"
    description: str = (
        "Browse a web URL using a headless browser and extract readable page text. "
        "Useful for reading articles, documentation, or websites."
    )
    is_concurrent_safe: bool = True
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The HTTP or HTTPS URL of the web page to read.",
            }
        },
        "required": ["url"],
    }

    async def execute(self, url: str, **kwargs: Any) -> str:
        return await browse_url(url)
