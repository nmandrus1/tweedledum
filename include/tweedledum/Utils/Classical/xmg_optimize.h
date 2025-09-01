#include <mockturtle/networks/xmg.hpp>
#include <mockturtle/algorithms/xmg_optimization.hpp>
#include <mockturtle/algorithms/xmg_algebraic_rewriting.hpp>
#include <mockturtle/algorithms/xmg_resub.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/xmg_npn.hpp>
#include <mockturtle/algorithms/cleanup.hpp>

inline void xmg_optimize(mockturtle::xmg_network& xmg){
  
  // Example 1: Apply Don't Care Optimization
  xmg = mockturtle::xmg_dont_cares_optimization(xmg);
  xmg = mockturtle::cleanup_dangling(xmg);

  // Example 2: Apply Algebraic Depth Rewriting
  mockturtle::xmg_algebraic_depth_rewriting_params ard_ps;
  ard_ps.strategy = mockturtle::xmg_algebraic_depth_rewriting_params::aggressive;
  // mockturtle::xmg_algebraic_depth_rewriting(xmg, ard_ps);
  xmg = mockturtle::cleanup_dangling(xmg);

  // Example 3: Apply XMG Resubstitution
  mockturtle::resubstitution_params resub_ps;
  // Configure resub_ps if needed (e.g., .max_divisors, .max_inserts)
  mockturtle::xmg_resubstitution(xmg, resub_ps);
  xmg = mockturtle::cleanup_dangling(xmg);

  // Example 4: Apply Cut Rewriting with XMG NPN Resynthesis
  mockturtle::cut_rewriting_params cr_ps;
  // Configure cr_ps (e.g., .cut_enumeration_ps.cut_size)
  // mockturtle::xmg_npn_resynthesis<mockturtle::xmg_network> xmg_resyn_fn;
  mockturtle::xmg_npn_resynthesis xmg_resyn_fn;
  mockturtle::cut_rewriting(xmg, xmg_resyn_fn, cr_ps);
  xmg = mockturtle::cleanup_dangling(xmg);

  // You can chain these operations or use them selectively.
  // The order and number of iterations can matter, similar to synthesis scripts in tools like ABC.
}
