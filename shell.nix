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
    
    # Use conda's Python if conda is activated
    if [ -n "$CONDA_PREFIX" ]; then
      export PATH="$CONDA_PREFIX/bin:$PATH"
      export PYTHONPATH="$CONDA_PREFIX/lib/python*/site-packages:$PYTHONPATH"
    fi
    
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
