/*------------------------------------------------------------------------------
| Part of Tweedledum Project.  This file is distributed under the MIT License.
| See accompanying file /LICENSE for details.
*-----------------------------------------------------------------------------*/
#include <pybind11/pybind11.h>
#include <string>
#include <tweedledum/Utils/Cut.h>
#include <tweedledum/Utils/LinPhasePoly.h>
#include <mockturtle/io/write_dot.hpp>
#include <mockturtle/networks/xag.hpp>

#include <string>

void init_utils(pybind11::module& module)
{
    using namespace tweedledum;
    namespace py = pybind11;

    py::class_<LinPhasePoly>(module, "LinPhasePoly")
        .def(py::init<>())
        .def("add_term", py::overload_cast<uint32_t, double const>(&LinPhasePoly::add_term));
        // This is too slow and I have no idea why :(
        // .def("add_term", py::overload_cast<std::vector<uint32_t> const&, double const>(&LinPhasePoly::add_term));

    py::class_<Cut>(module, "Cut")
        .def("cbits", &Cut::py_cbits)
        .def("qubits", &Cut::py_qubits)
        .def("instructions", &Cut::py_instructions);

    // module.def("xag_export_dot",
    //            &mockturtle::write_dot<mockturtle::xag_network>,
    //            py::arg("xag"),
    //            py::arg("filename"),
    //            "Export the XAG in DOT format to a file");

    // Use a lambda to wrap the call to mockturtle::write_dot
    module.def("xag_export_dot",
               // The lambda takes the arguments you expose to Python
               [](mockturtle::xag_network const& xag, std::string const& filename) {
                   // Call the C++ function. C++ will handle the default argument
                   // for the 'drawer' parameter automatically.
                   mockturtle::write_dot(xag, filename, mockturtle::gate_dot_drawer<mockturtle::xag_network>{});
               },
               py::arg("xag"),           // Argument 1 for the lambda
               py::arg("filename"),     // Argument 2 for the lambda
               "Export the XAG in DOT format to a file"); // Docstring
}
