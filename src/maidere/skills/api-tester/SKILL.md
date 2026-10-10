---
name: api-tester
description: Testing and validating REST and FastAPI endpoints using httpx or curl.
triggers:
  - api
  - endpoint
  - curl
  - http
  - fastapi
  - test
  - rest
---

# API Testing & Endpoint Validation

## Instructions
1. Verify the target API endpoint specification before crafting requests.
2. Test both happy-path (200 OK) and edge-case status codes (400, 404, 500).
3. Inspect JSON response schema and headers for correct Content-Type.
4. Use async python scripts via `code_runner` or `shell` tool for automated tests.
5. Log HTTP status code and response payload preview clearly.
