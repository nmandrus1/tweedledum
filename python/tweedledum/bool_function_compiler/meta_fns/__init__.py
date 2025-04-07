from .binary_popcount_network import generate_binary_popcount
from .sorting_network import generate_sorting_network

_global_generators = {
    "generate_sorting_network": generate_sorting_network,
    # "generate_sorting_network_fixed": generate_sorting_network_fixed,
    "generate_binary_popcount": generate_binary_popcount,
}
