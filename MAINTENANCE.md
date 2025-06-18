# Maintenance Plan for Balsa

**BALSA** is a comprehensive benchmark suite for evaluating active learning and optimisation algorithms in real-world scientific design tasks.

## Scope
This project provides a suite of benchmark problems including real-world scientific tasks (electron ptychography, protein design), synthetic test functions, and evaluation tools for comparing optimisation algorithms. It implements 10+ optimisation methods with standardized evaluation protocols. The current feature set is stable and functional.

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