---
name: git-workflow
description: Professional Git operations, conventional commits, and merge safety checks.
triggers:
  - git
  - commit
  - branch
  - merge
  - rebase
  - pull
  - push
---

# Git Workflow & Conventional Commits

## Instructions
1. Always run `git status` via the shell tool before staging changes.
2. Format all commit messages using Conventional Commits: `<type>(<scope>): <subject>`.
3. Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
4. Never stage `.env` or sensitive database files (`db/*.db`).
5. Ensure clean branch status before attempting merges or rebases.
