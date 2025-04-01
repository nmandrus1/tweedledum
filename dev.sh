#!/bin/bash
set -e  # Exit on error

function help() {
  echo "Tweedledum Development Commands"
  echo "------------------------------"
  echo "Usage: ./dev.sh COMMAND"
  echo ""
  echo "Commands:"
  echo "  build      - Build the project in development mode"
  echo "  debug      - Build with debug symbols"
  echo "  test       - Run the test suite"
  echo "  clean      - Remove build artifacts"
  echo "  help       - Show this help message"
}

# Enter nix-shell if not already in it
if [ -z "$IN_NIX_SHELL" ]; then
  echo "Entering nix-shell..."
  exec nix-shell --run "bash $0 $*"
fi

# Command handling
case "$1" in
  build)
    echo "Building in development mode..."
    CMAKE_ARGS="-DTWEEDLEDUM_PYBINDS=ON -DTWEEDLEDUM_EXAMPLES=OFF -DTWEEDLEDUM_TESTS=OFF" \
    pip install -e .
    ;;
    
  debug)
    echo "Building with debug symbols..."
    CMAKE_ARGS="-DTWEEDLEDUM_PYBINDS=ON -DTWEEDLEDUM_EXAMPLES=OFF -DTWEEDLEDUM_TESTS=OFF -DCMAKE_BUILD_TYPE=Debug" \
    pip install -e .
    ;;
    
  test)
    echo "Running tests..."
    pytest ${@:2}
    ;;
    
  clean)
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/
    find python -name "*.so" -delete
    ;;
    
  help|"")
    help
    ;;
    
  *)
    echo "Unknown command: $1"
    help
    exit 1
    ;;
esac
