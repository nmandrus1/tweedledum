from .binary_popcount_network import generate_binary_popcount
from .sorting_network import generate_sorting_network
from .batcher_sort import generate_batcher_sort_network

_global_generators = {
    "generate_sorting_network": generate_sorting_network,
    "generate_batcher_sort_network": generate_batcher_sort_network,
    "generate_binary_popcount": generate_binary_popcount,
}
