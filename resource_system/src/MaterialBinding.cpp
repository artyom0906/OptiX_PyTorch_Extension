#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "../include/IMaterialSystem.h"
#include "../include/MaterialSystem.h"

namespace py = pybind11;
using namespace optix_renderer;

// Python bindings for the material system
namespace optix_renderer {
void bind_material_system(py::module& m) {
    // Bind material type enum
    py::enum_<OptiXMaterialType>(m, "MaterialType")
        .value("LAMBERTIAN", OPTIX_LAMBERTIAN)
        .value("PBR", OPTIX_PBR)
        .value("GLASS", OPTIX_GLASS)
        .value("EMISSIVE", OPTIX_EMISSIVE)
        .value("MIRROR", OPTIX_MIRROR)
        .value("MDL", OPTIX_MDL);
    
    // Bind material flags enum
    py::enum_<MaterialFlags>(m, "MaterialFlags")
        .value("HAS_ALBEDO_TEXTURE", MATERIAL_HAS_ALBEDO_TEXTURE)
        .value("HAS_NORMAL_TEXTURE", MATERIAL_HAS_NORMAL_TEXTURE)
        .value("HAS_METALLIC_ROUGHNESS_TEXTURE", MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE)
        .value("HAS_EMISSION_TEXTURE", MATERIAL_HAS_EMISSION_TEXTURE)
        .value("IS_EMISSIVE", MATERIAL_IS_EMISSIVE)
        .value("IS_TRANSPARENT", MATERIAL_IS_TRANSPARENT)
        .value("HAS_ANISOTROPY", MATERIAL_HAS_ANISOTROPY)
        .value("HAS_CLEARCOAT", MATERIAL_HAS_CLEARCOAT)
        .value("HAS_SHEEN", MATERIAL_HAS_SHEEN)
        .value("IS_MDL", MATERIAL_IS_MDL);
    
    // Bind the Material class
    py::class_<MaterialSystem, std::shared_ptr<MaterialSystem>>(m, "MaterialSystem")
        .def(py::init<>())
        .def("initialize", &MaterialSystem::initialize)
        .def("load_material_from_file", &MaterialSystem::loadMaterialFromFile,
             py::arg("filename"), py::arg("material_name"))
        .def("create_material_instance", &MaterialSystem::createMaterialInstance,
             py::arg("material_desc"))
        .def("set_parameter", [](MaterialSystem& self, MaterialHandle material, 
                              const std::string& paramName, py::object value) {
            // Handle different parameter types
            if (py::isinstance<py::float_>(value)) {
                return self.setParameter(material, paramName, value.cast<float>());
            } else if (py::isinstance<py::int_>(value)) {
                return self.setParameter(material, paramName, value.cast<int>());
            } else if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                // Convert list/tuple to vector<float> for float3 parameters
                std::vector<float> vec;
                for (auto item : value) {
                    vec.push_back(item.cast<float>());
                }
                return self.setParameter(material, paramName, vec);
            } else {
                throw std::runtime_error("Unsupported parameter type");
            }
        }, py::arg("material"), py::arg("param_name"), py::arg("value"))
        .def("set_texture_parameter", &MaterialSystem::setTextureParameter,
             py::arg("material"), py::arg("param_name"), py::arg("texture_handle"));
    
    // Helper function to create default material (needs to be defined before use)
    auto create_default_material = [](const std::shared_ptr<MaterialSystem>& mat_system, 
                                     OptiXMaterialType type) -> MaterialHandle {
        if (!mat_system->initialize()) {
            throw std::runtime_error("Failed to initialize MaterialSystem");
        }
        
        // Create description and instance based on type
        MaterialHandle description = 0;
        
        // For now, we use fixed handles for the standard materials created during initialization
        switch (type) {
            case OPTIX_LAMBERTIAN: description = 1; break;
            case OPTIX_PBR: description = 2; break;
            case OPTIX_GLASS: description = 3; break;
            case OPTIX_EMISSIVE: description = 4; break;
            case OPTIX_MIRROR: description = 5; break;
            default: throw std::runtime_error("Unsupported material type");
        }
        
        return mat_system->createMaterialInstance(description);
    };

    // Add factory function to create a default material
    m.def("create_default_material", create_default_material, 
          py::arg("material_system"), py::arg("type"));
    
    // Add utility function to create a colored material with default parameters
    m.def("create_colored_material", [create_default_material](
                                     const std::shared_ptr<MaterialSystem>& mat_system,
                                     OptiXMaterialType type, 
                                     const std::vector<float>& color) -> MaterialHandle {
        if (!mat_system->initialize()) {
            throw std::runtime_error("Failed to initialize MaterialSystem");
        }
        
        // Create the material
        MaterialHandle handle = create_default_material(mat_system, type);
        
        // Set color
        if (type == OPTIX_EMISSIVE) {
            mat_system->setParameter(handle, "emission", color);
        } else {
            mat_system->setParameter(handle, "albedo", color);
        }
        
        return handle;
    }, py::arg("material_system"), py::arg("type"), py::arg("color"));
}
} // namespace optix_renderer