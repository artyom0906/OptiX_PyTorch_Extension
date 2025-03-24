//
// Created by artem on 2/28/25.
//
#include "../include/PythonBindings.h"
#include <optix_function_table_definition.h>
namespace optix_renderer{
    // Function to create the Python module
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.doc() = "OptiX renderer with resource management system";
        initPythonBindings(m);
    }
}
