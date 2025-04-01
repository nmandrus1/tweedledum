{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # C++ build tools
    pkgs.cmake
    pkgs.ninja
    pkgs.gcc11
    pkgs.pkg-config
    
    # Libraries
    pkgs.eigen
    
    # Python build tools - avoid specific Python packages that might cause conflicts
    pkgs.python310
  ];

  # Environment setup
  shellHook = ''
    # Set up C++ compiler
    export CC="${pkgs.gcc11}/bin/gcc"
    export CXX="${pkgs.gcc11}/bin/g++"
    
    # Configure cmake arguments
    export CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"
    
    # Add project root to PYTHONPATH
    export PYTHONPATH=$PWD/python:$PYTHONPATH
    
    # If conda exists, activate environment
    if command -v conda &> /dev/null; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate tweedledum-dev 2>/dev/null || echo "Run: conda create -n tweedledum-dev python=3.10"
    fi
    
    echo "Tweedledum Development Environment"
    echo "=================================="
    echo ""
    echo "Development commands:"
    echo "  - Install in development mode:  pip install -e ."
    echo "  - Build with debug symbols:     pip install -e . --config-settings=cmake.build-type=Debug"
    echo "  - Run tests:                    pytest"
    echo ""
  '';
}
