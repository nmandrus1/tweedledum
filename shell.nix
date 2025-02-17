{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.cmake
    pkgs.ninja
    pkgs.gcc11  # Use GCC 11
    pkgs.pkg-config
    pkgs.eigen
  ];

  cc = "${pkgs.gcc11}/bin/gcc";
  cxx = "${pkgs.gcc11}/bin/g++";

  # Use shellHook to set CC and CXX, and to inform the user.
  shellHook = ''
    export CC="${pkgs.gcc11}/bin/gcc"
    export CXX="${pkgs.gcc11}/bin/g++"
    echo "Using GCC 11:"
    g++ --version
    echo "Make sure your conda environment is activated:"
    echo "  conda activate tweedledum-dev"
    echo "Then run 'pip install -U pip setuptools wheel' and 'pip install -e \".[dev]\" --no-build-isolation --no-deps'"
  '';
}
