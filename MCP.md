# Haiqu MCP Server Setup

Haiqu exposes two MCP (Model Context Protocol) servers:

| Server | URL | Purpose |
|--------|-----|---------|
| **Haiqu API** | `https://api.haiqu.ai/mcp` | Execute circuits, access SDK tools via AI assistants |
| **Haiqu Docs** | `https://docs.haiqu.ai/mcp` | Query Haiqu documentation from your AI assistant |

Replace `HAIQU-API-KEY` with your actual API key in all configurations below.

---

## VS Code + Claude Code

Add to `~/.claude.json`:

```json
"mcpServers": {
  "Haiqu": {
    "type": "http",
    "url": "https://api.haiqu.ai/mcp",
    "headers": {
      "Authorization": "HAIQU-API-KEY"
    }
  },
  "HaiquDocumentation": {
    "url": "https://docs.haiqu.ai/mcp"
  }
}
```

---

## Cursor

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "Haiqu": {
      "url": "https://api.haiqu.ai/mcp",
      "type": "http",
      "headers": {
        "authorization": "HAIQU-API-KEY"
      }
    },
    "HaiquDocumentation": {
      "url": "https://docs.haiqu.ai/mcp"
    }
  }
}
```

---

Full AI-assisted development guide: [docs.haiqu.ai/quickstart/ai-assisted-development](https://docs.haiqu.ai/quickstart/ai-assisted-development)
