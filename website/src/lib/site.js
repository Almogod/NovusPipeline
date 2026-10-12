export const GITHUB_URL = 'https://github.com/Almogod/NovusPipeline'
export const PYPI_URL = 'https://pypi.org/project/novuspipeline/'
export const CHANGELOG_URL = `${GITHUB_URL}/blob/main/CHANGELOG.md`
export const SECURITY_URL = `${GITHUB_URL}/blob/main/SECURITY.md`
export const DOCS_URL = `${GITHUB_URL}#readme`

export const UVX_COMMAND = 'uvx novuspipeline'

// Exactly as documented in the README ("Connect your MCP client").
export const CLIENT_SETUPS = [
  {
    id: 'claude-code',
    label: 'Claude Code',
    language: 'shell',
    code: 'claude mcp add novuspipeline -- uvx --from "novuspipeline[gnn]" novuspipeline --workspace /path/to/project',
  },
  {
    id: 'json',
    label: 'Claude Desktop & Cursor',
    language: 'json',
    code: `{
  "mcpServers": {
    "novuspipeline": {
      "command": "uvx",
      "args": ["--from", "novuspipeline[gnn]", "novuspipeline", "--workspace", "/absolute/path/to/your/project"]
    }
  }
}`,
  },
  {
    id: 'pip',
    label: 'pip',
    language: 'shell',
    code: 'pip install "novuspipeline[gnn]"',
  },
]
