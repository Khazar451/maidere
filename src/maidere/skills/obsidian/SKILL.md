---
name: obsidian
description: Create, read, organize, search, and open structured markdown notes in the local Obsidian Vault.
triggers:
  - obsidian
  - vault
  - markdown note
  - journal
  - daily note
  - knowledge base
  - write note
  - open note
  - search notes
---

# Obsidian Vault Note Protocol

You have direct access to the user's local **Obsidian Vault** via the `obsidian` tool.

## Obsidian Markdown Guidelines

1. **YAML Frontmatter (Standard):**
   Every note in Obsidian should include frontmatter metadata at the top:
   ```markdown
   ---
   title: "Descriptive Note Title"
   date: YYYY-MM-DD HH:MM
   tags:
     - topic/subtopic
     - research
   created_by: Maidere
   ---
   ```

2. **Obsidian Callouts:**
   Use GitHub / Obsidian flavored callouts for emphasis:
   - `> [!SUMMARY] Core Takeaway`
   - `> [!NOTE] Implementation Detail`
   - `> [!TIP] Best Practice`
   - `> [!WARNING] Critical Bottleneck`
   - `> [!INFO] Reference Info`

3. **Bidirectional Wikilinks & Tags:**
   - Link related topics using `[[Note Name]]` or `[[Folder/Note Name|Alias]]`.
   - Organize concepts with nested tags like `#ai/llm` or `#engineering/python`.

4. **Available Tool Actions:**
   - `obsidian(action="write_note", title="Title", content="...", folder="OptionalSubfolder", tags=["tag1", "tag2"])`
   - `obsidian(action="read_note", title="Title")`
   - `obsidian(action="list_notes", folder="OptionalSubfolder")`
   - `obsidian(action="search_notes", query="search term")`
   - `obsidian(action="open_note", title="Title")` — launches the desktop Obsidian app and opens the note!
