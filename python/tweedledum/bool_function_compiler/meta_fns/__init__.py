from .sorting_network import generate_sorting_network
from .binary_popcount_network import generate_binary_popcount

_global_generators = {
    "generate_sorting_network": generate_sorting_network,
    "generate_binary_popcount": generate_binary_popcount,
}
