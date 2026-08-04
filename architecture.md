# NovusPipeline Architecture Blueprint

- **Server Protocol**: Model Context Protocol (MCP) via Stdio transport (`fastmcp` 3.4.5).
- **Runtime**: Python 3.11+.
- **Vector DB**: ChromaDB for local embedding storage & query retrieval.
- **Git Integration**: GitPython for staging, branch creation, and PR drafting.
- **Verification Engine**: Subprocess execution sandbox with command whitelisting.
