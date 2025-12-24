# Contributing to Goeckoh

Thank you for your interest in contributing to Goeckoh! This guide will help you get started.

## Repository Organization

Please familiarize yourself with the repository structure documented in [README.md](README.md):

- `/docs/` - All documentation (see [docs/INDEX.md](docs/INDEX.md))
- `/config/` - Configuration files (see [config/README.md](config/README.md))
- `/scripts/` - Build and deployment scripts (see [scripts/README.md](scripts/README.md))
- `/archive/` - Archived materials (see [archive/README.md](archive/README.md))
- `/website/` - Marketing website
- `/cognitive-nebula/` - 3D visualization frontend
- `/GOECKOH/` - Main application package
- `/mobile/` - Mobile platform code

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaleidoscopeAI/goeckoh-site.git
   cd goeckoh-site
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure the system**
   ```bash
   # Configuration is in the config/ directory
   python -m cli validate
   ```

4. **Read the documentation**
   - Start with [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)
   - Review [docs/system/SYSTEM_OVERVIEW.md](docs/system/SYSTEM_OVERVIEW.md)

## Making Changes

### File Organization Rules

1. **Keep files organized** in the appropriate directory:
   - Documentation → `/docs/`
   - Configuration → `/config/`
   - Build scripts → `/scripts/`
   - Research/reference → `/archive/research/`
   - Old versions → `/archive/old-docs/`

2. **Don't clutter the root directory** - files should be in appropriate subdirectories

3. **Update documentation** when changing structure or functionality

### Code Guidelines

1. **Follow existing patterns** in the codebase
2. **Add comments** for complex logic
3. **Test your changes** before submitting
4. **Update configuration** if adding new features that need config options

### Documentation Guidelines

1. **Keep documentation current** - update docs when code changes
2. **Use clear headings** and proper markdown formatting
3. **Add examples** where helpful
4. **Link to related docs** using relative paths

### Configuration Changes

1. **Update schema** if modifying `config/config.yaml` structure
2. **Maintain backward compatibility** when possible
3. **Document new options** in `config/README.md`
4. **Test validation** with `python -m cli validate`

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the file organization rules above
   - Keep commits focused and descriptive

3. **Test your changes**
   - Run existing tests
   - Verify configuration still loads
   - Check that builds still work

4. **Update documentation**
   - Update relevant README files
   - Add or update docs in `/docs/` as needed
   - Update the main README.md if structure changes

5. **Submit pull request**
   - Provide clear description of changes
   - Reference any related issues
   - Explain why the change is needed

## Questions?

- Check [docs/INDEX.md](docs/INDEX.md) for documentation
- Review [README.md](README.md) for repository overview
- File an issue on GitHub for questions or bugs

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project and community
- Show empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to Goeckoh! 🎯
