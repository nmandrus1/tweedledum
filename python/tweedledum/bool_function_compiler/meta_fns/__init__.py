from .batcher_sort import generate_batcher_sort_network
from .binary_popcount_network import generate_binary_popcount
from .cardinality import generate_at_least_k_counter, generate_exactly_k_counter
from .sorting_network import generate_sorting_network

_global_generators = {
    "generate_sorting_network": generate_sorting_network,
    "generate_batcher_sort_network": generate_batcher_sort_network,
    "generate_binary_popcount": generate_binary_popcount,
    "generate_at_least_k_counter": generate_at_least_k_counter,
    "generate_exactly_k_counter": generate_exactly_k_counter,
}
