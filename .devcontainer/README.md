# HyperView DevContainer

This directory contains the development container configuration for HyperView. The devcontainer provides a consistent, reproducible development environment that works identically on **macOS (ARM64)**, **Linux (AMD64)**, and **GitHub Codespaces**.

## What's Included

### Base Environment
- **Python 3.11** with all ML/image processing system dependencies
- **Node.js 22** with npm for frontend development
- **uv** package manager for fast Python dependency management
- **Git** and **GitHub CLI** for version control

### Pre-installed Tools
- **Ruff**: Python linting and formatting
- **pytest**: Python testing framework
- **pre-commit**: Git hooks for code quality
- **ESLint**: Frontend linting
- **All project dependencies**: Python packages and npm packages

### VSCode Extensions
- **Python**: ms-python.python, ms-python.vscode-pylance
- **Ruff**: charliermarsh.ruff
- **TypeScript/JavaScript**: dbaeumer.vscode-eslint, esbenp.prettier-vscode
- **Tailwind CSS**: bradlc.vscode-tailwindcss
- **Git**: eamodio.gitlens
- **GitHub Copilot**: GitHub.copilot (optional)

### Port Forwarding
- **Port 5151**: HyperView Backend (FastAPI)
- **Port 3000**: Frontend Dev Server (Next.js)

## Quick Start

### Option 1: Using VSCode (Local)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [VSCode](https://code.visualstudio.com/)
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
4. Open this repository in VSCode
5. Click the green button in the bottom-left corner and select **"Reopen in Container"**
6. Wait for the container to build and setup to complete (~5-10 minutes first time)
7. Start developing!

### Option 2: Using GitHub Codespaces

1. Go to the [HyperView repository](https://github.com/your-org/HyperView) on GitHub
2. Click the green **"Code"** button
3. Select the **"Codespaces"** tab
4. Click **"Create codespace on main"**
5. Wait for the environment to initialize (~5-10 minutes first time)
6. Start developing in the browser or connect with VSCode!

## Architecture

### File Structure

```
.devcontainer/
├── devcontainer.json          # Main configuration
├── Dockerfile                 # Container image definition
├── postCreateCommand.sh       # Setup script (runs after container creation)
└── README.md                  # This file
```

### Build Process

1. **Base Image**: `mcr.microsoft.com/devcontainers/python:1-3.11-bullseye`
   - Official Microsoft devcontainer image
   - Multi-platform support (AMD64 and ARM64)
   - Pre-configured for Python development

2. **System Dependencies** (installed in Dockerfile):
   - Build tools: `build-essential`, `git`, `curl`
   - CLIP/ML dependencies: `libssl-dev`, `pkg-config`
   - Image processing: `libjpeg-dev`, `libpng-dev`, `libtiff-dev`
   - NumPy acceleration: `libopenblas-dev`, `liblapack-dev`

3. **Runtime Setup** (postCreateCommand.sh):
   - Install uv package manager
   - Create Python virtual environment (`.venv`)
   - Install Python dependencies with `uv pip install -e ".[dev]"`
   - Install pre-commit hooks
   - Install frontend dependencies with `npm install`

## Platform Compatibility

| Component | macOS (ARM64) | Codespaces (AMD64) | Notes |
|-----------|---------------|--------------------|-------|
| Base Image | ✅ | ✅ | Multi-arch support |
| Python 3.11 | ✅ | ✅ | Official builds |
| Node.js 22 | ✅ | ✅ | Via devcontainer feature |
| uv installer | ✅ | ✅ | Auto-detects platform |
| System deps | ✅ | ✅ | Platform-agnostic packages |
| Docker Desktop | ✅ Required | ❌ Not needed | Only for local development |

## Development Workflow

### Running the Demo

```bash
# Quick demo with 500 samples
hyperview demo --samples 500

# Opens http://localhost:5151
```

### Development Mode (Hot Reload)

**Terminal 1 - Backend:**
```bash
python scripts/demo.py --samples 200 --no-browser
# Backend API: http://localhost:5151
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# Frontend: http://localhost:3000 (proxies to backend)
```

### Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Python linting
ruff check src/

# Python formatting
ruff format src/

# Frontend linting
cd frontend && npm run lint
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hyperview
```

## Configuration Details

### Environment Variables

- `PYTHONUNBUFFERED=1`: Disable Python output buffering
- `UV_CACHE_DIR=/workspaces/HyperView/.cache/uv`: Cache uv downloads

### VSCode Settings

- **Python**: Auto-activate `.venv`, use Ruff for formatting
- **TypeScript**: Use Prettier, format on save
- **Editor**: 100-character ruler (matches Ruff config)
- **Git**: Auto-fetch, smart commit

### Port Forwarding

Ports are automatically forwarded and accessible at:
- **Local**: `http://localhost:<port>`
- **Codespaces**: `https://<codespace-name>-<port>.preview.app.github.dev`

## Customization

### Adding Python Dependencies

```bash
# Add to pyproject.toml, then:
uv pip install -e ".[dev]"
```

### Adding VSCode Extensions

Edit `.devcontainer/devcontainer.json`:
```json
"customizations": {
  "vscode": {
    "extensions": [
      "publisher.extension-name"
    ]
  }
}
```

Then rebuild the container: **Cmd/Ctrl+Shift+P** → **"Rebuild Container"**

### Modifying System Dependencies

Edit `.devcontainer/Dockerfile` to add `apt-get` packages, then rebuild.

## Troubleshooting

### Container Build Fails

**Issue**: Build fails with dependency errors

**Solution**:
```bash
# Rebuild without cache
Cmd/Ctrl+Shift+P → "Rebuild Container Without Cache"
```

### Port Already in Use

**Issue**: Port 5151 or 3000 already in use

**Solution**:
```bash
# Use different ports
hyperview demo --port 5152
cd frontend && PORT=3001 npm run dev
```

### uv Installation Fails

**Issue**: `curl: command not found` or network error

**Solution**:
- Check internet connection
- Rebuild container (curl is pre-installed in Dockerfile)

### Pre-commit Hooks Fail

**Issue**: Frontend lint fails with "next: command not found"

**Solution**:
```bash
# Reinstall frontend dependencies
cd frontend && npm install
```

### Python Virtual Environment Not Activated

**Issue**: Commands like `hyperview` not found

**Solution**:
```bash
# Manually activate
source .venv/bin/activate

# Or restart terminal (auto-activation in bashrc)
```

## Performance Tips

### Faster Rebuilds

The devcontainer uses volume mounts for caching:
- `.cache/uv`: uv package cache (persisted across rebuilds)

To clear caches:
```bash
rm -rf .cache
```

### Codespaces Prebuilds

For faster Codespaces startup, enable prebuilds in your repository settings:
1. Go to **Settings** → **Codespaces**
2. Enable **"Prebuild"** for the main branch
3. Prebuilds will automatically run on push

### Local Performance (macOS)

Docker Desktop on macOS can be slower than native. For best performance:
- Allocate more resources: **Docker Desktop** → **Settings** → **Resources**
- Use VirtioFS (enabled by default in newer versions)
- Consider developing natively if devcontainer is too slow

## Security

### Non-root User

The container runs as the `vscode` user (non-root) for security.

### Credential Forwarding

- **Git credentials**: Automatically forwarded from host
- **SSH keys**: Forwarded via SSH agent
- **GitHub CLI**: Use `gh auth login` in container

### Secrets in Codespaces

Use [GitHub Codespaces secrets](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-secrets-for-your-codespaces) for sensitive data.

## CI/CD Integration

This devcontainer can be used in CI/CD:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: mcr.microsoft.com/devcontainers/python:1-3.11-bullseye
    steps:
      - uses: actions/checkout@v3
      - run: bash .devcontainer/postCreateCommand.sh
      - run: pytest
```

## Additional Resources

- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [HyperView Main README](../README.md)
- [HyperView Development Guide](../CLAUDE.md)

## Support

For issues with the devcontainer:
1. Check this README for troubleshooting steps
2. Rebuild the container without cache
3. Check [Docker Desktop status](https://www.docker.com/products/docker-desktop/)
4. Open an issue on GitHub

---

**Happy coding!** 🚀
