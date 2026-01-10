# CUDA Setup Instructions

To enable CUDA support for PyTorch in this project, we have explicitly configured Poetry to use the official PyTorch wheel source (CUDA 12.4).

## Steps to Apply Changes

1.  **Update Lock File**:
    Run the following command to regenerate the lock file with the new source configuration:
    ```powershell
    poetry lock
    ```

2.  **Re-install Dependencies**:
    Install the packages from the new source. The `--sync` flag ensures unrelated packages are removed if needed, but a standard install is usually sufficient.
    ```powershell
    poetry install --sync
    ```
    *Note: This may download a large file (~2.5GB) for the CUDA-enabled PyTorch wheel.*

3.  **Verify Installation**:
    Check if CUDA is available by running:
    ```powershell
    poetry run python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    ```
    You should see `CUDA available: True`.

## Troubleshooting

-   **"No compatible version found"**: If Poetry cannot find version `2.9.1` in the `pytorch` source, try relaxing the version constraint in `pyproject.toml` (e.g., `torch = { version = ">=2.5.0", source = "pytorch" }`).
-   **Driver Mismatch**: Ensure your NVIDIA drivers are up to date. You have CUDA 12.9 drivers, which should be compatible with the CUDA 12.4 runtime used by PyTorch.
