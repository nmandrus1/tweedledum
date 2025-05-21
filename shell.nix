{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # C++ build tools
    pkgs.cmake
    pkgs.ninja
    pkgs.gcc11
    pkgs.pkg-config
    pkgs.eigen
  ];

  shellHook = ''
    # Set up C++ compiler
    export CC="${pkgs.gcc11}/bin/gcc"
    export CXX="${pkgs.gcc11}/bin/g++"
    
    # Configure cmake arguments
    export CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DTWEEDLEDUM_PYBINDS=ON -DTWEEDLEDUM_EXAMPLES=OFF -DTWEEDLEDUM_TESTS=OFF"
   
    echo "Entered Nix shell for tweedledum library..."
    # Replace 'bash' with your shell (e.g., zsh) if needed
    eval "$(conda shell.bash hook)"

    # Use the CONDA_ENV_NAME variable inherited from direnv/calling shell.
    # If CONDA_ENV_NAME is not set or empty, default to "tweedledum".
    # ESCAPE the $ for Nix evaluation using ''$
    local target_conda_env=''${CONDA_ENV_NAME:-tweedledum}

    # Using printf is slightly safer than echo for printing variables
    printf "Activating Conda environment: %s\n" "$target_conda_env"
    
    # Use quotes around the variable for safety
    conda activate "$target_conda_env"

    echo "Nix shell setup complete with Conda env active."
    
    echo "Tweedledum C++ Development Environment Ready"
    echo "-------------------------------------------"
    echo "GCC: $(gcc --version | head -n1)"
    echo "Python: $(python --version)"
    echo ""
    echo "To build in development mode:"
    echo "  pip install -e ."
    echo ""
  '';
}
