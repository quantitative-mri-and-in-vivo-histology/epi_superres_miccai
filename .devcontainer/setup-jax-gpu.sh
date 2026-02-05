#!/bin/bash
# Auto-detect GPU and install appropriate JAX version

# Check if GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    # Check if JAX CUDA is already installed
    if python -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "cuda"; then
        echo "JAX CUDA already installed and working"
    else
        echo "GPU detected, installing JAX CUDA..."
        pip install -U "jax[cuda12]" -q
        echo "JAX CUDA installed. Devices available:"
        python -c "import jax; print(jax.devices())"
    fi
else
    echo "No GPU detected, using CPU JAX"
fi
