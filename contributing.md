### Development Environment Setup

We use `uv` for fast and reliable dependency management. Follow these steps to set up your environment for contributing.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/path/to/speaches.git
    cd speaches
    ```

2.  **Create and Activate a Virtual Environment:**
    Using a virtual environment is essential for isolating project dependencies.
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync --all-extras --upgrade
    ```

3.  **Install All Dependencies in Editable Mode:**
    The following command installs the `speaches` package itself, plus all optional dependencies required for development and running the full test suite. The `-e` flag (for "editable") links the installation to your source code, so you don't need to reinstall after making changes.
    ```bash
    uv pip install -e '.[dev]'
    ```

You are now set up for development. You can run the server with `speaches serve` and run the test suite with `pytest`.