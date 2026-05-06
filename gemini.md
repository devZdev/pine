# Core Coding Standards for Pine Script Repository

This repository follows strict coding standards to ensure all indicators remain clean, maintainable, and highly reusable. When generating or modifying scripts in this repository, the following rules MUST be adhered to:

1. **Version:** All scripts must use Pine Script `//@version=5`.
2. **Code Styling:** Strict adherence to excellent code styling is required. Use clear, self-documenting variable names (e.g., `baselineMA`, `maLengthInput` instead of `ma`, `len`).
3. **Architecture:** Favor modular, user-defined functions for core mathematical calculations. This ensures complex logic is isolated and reusable across different indicators in the future.
4. **Inputs:** Group all `input.*` calls logically at the absolute top of the script, immediately following the `indicator` or `strategy` declaration. Use `group` parameters within inputs to organize the UI in TradingView.