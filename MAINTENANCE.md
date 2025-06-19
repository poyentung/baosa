# Maintenance Plan for Balsa

## Maintenance Policy
- **Bug Reports**: Major issues will be reviewed and, if feasible, fixed.
- **Contributions**: Community pull requests are welcome. We will review:
  - Fixes for critical bugs or dependency issues.
  - Additions of new benchmark tasks (e.g. additional scientific optimisation problems, synthetic functions).
  - New optimisation algorithms or improvements to existing implementations.

## Planned Future Development
- **Tool Integration**: Integration of existing gradient-free optimisation libraries like [Nevergrad](https://github.com/facebookresearch/nevergrad) to expand the available algorithms and improve benchmarking capabilities.

## Contributing
To contribute:
1. Open an issue or pull request with a clear description.
2. Run `ruff check . --fix && ruff format .` to ensure code quality and formatting.
3. Keep additions modular and aligned with the existing framework.
4. Include relevant tests or examples if possible.

---

Maintainer: [@poyentung](https://github.com/poyentung)