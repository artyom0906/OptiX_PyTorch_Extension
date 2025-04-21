#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ResourceManager.cuh"
#include "GeometryInstance.h"
#include "Renderer.h"
#include "OpenVRSystem.cuh"

namespace py = pybind11;

namespace optix_renderer {

// Forward declaration of material system binding
void bind_material_system(py::module& m);

// Python bindings for the resource management system
void initPythonBindings(py::module& m) {

    py::class_<OpenVRSystem>(m, "OpenVRSystem")
            .def(py::init<ResourceManager*, int>(),
                 py::arg("resource_manager") = nullptr,
                 py::arg("device_id") = 0)
            .def("Initialize", &OpenVRSystem::Initialize)
            .def("Shutdown", &OpenVRSystem::Shutdown)
            .def("RenderFrame", &OpenVRSystem::RenderFrame)
            .def("ShouldClose", &OpenVRSystem::ShouldClose)
            .def("PollEvents", &OpenVRSystem::PollEvents)
            .def("SetRenderers", &OpenVRSystem::SetRenderers)
            .def("GetRecommendedRenderSize", &OpenVRSystem::GetRecommendedRenderSize)
            .def("GetEyeTransforms", [](OpenVRSystem& self) {
                std::vector<float> leftMat(12, 0.0f);  // 3x4 matrix
                std::vector<float> rightMat(12, 0.0f); // 3x4 matrix
                self.GetEyeTransforms(leftMat.data(), rightMat.data());
                return std::make_pair(leftMat, rightMat);
            })
            .def("GetEyeToHeadTransforms", [](OpenVRSystem& self) {
                std::vector<float> leftMat(12, 0.0f);  // 3x4 matrix
                std::vector<float> rightMat(12, 0.0f); // 3x4 matrix
                self.GetEyeToHeadTransforms(leftMat.data(), rightMat.data());
                return std::make_pair(leftMat, rightMat);
            })
            .def("GetHeadPoseMatrix", [](OpenVRSystem& self) {
                std::vector<float> headMat(12, 0.0f);  // 3x4 matrix
                self.GetHeadPoseMatrix(headMat.data());
                return headMat;
            })
            .def("GetProjectionMatrices", [](OpenVRSystem& self, float nearClip, float farClip) {
                std::vector<float> leftMat(16, 0.0f);  // 4x4 matrix
                std::vector<float> rightMat(16, 0.0f); // 4x4 matrix
                self.GetProjectionMatrices(leftMat.data(), rightMat.data(), nearClip, farClip);
                return std::make_pair(leftMat, rightMat);
            }, py::arg("near_clip") = 0.1f, py::arg("far_clip") = 100.0f)
            // Controller-related bindings
            .def("IsControllerValid", &OpenVRSystem::IsControllerValid)
            .def("GetControllerPose", [](OpenVRSystem& self, bool isLeft) {
                std::vector<float> poseMatrix(12, 0.0f);  // 3x4 matrix
                self.GetControllerPose(isLeft, poseMatrix.data());
                return poseMatrix;
            })
            .def("IsControllerTriggerPressed", &OpenVRSystem::IsControllerTriggerPressed)
            .def("IsControllerGripPressed", &OpenVRSystem::IsControllerGripPressed)
            .def("GetControllerTouchpadPosition", [](OpenVRSystem& self, bool isLeft) {
                float x = 0.0f, y = 0.0f;
                self.GetControllerTouchpadPosition(isLeft, &x, &y);
                return std::make_pair(x, y);
            })
            .def("IsControllerTouchpadPressed", &OpenVRSystem::IsControllerTouchpadPressed)
            .def("SetControllerInteractionEnabled", &OpenVRSystem::SetControllerInteractionEnabled)
            .def("IsControllerInteractionEnabled", &OpenVRSystem::IsControllerInteractionEnabled)
            .def("MoveObjectWithController", &OpenVRSystem::MoveObjectWithController);

    // Bind material system
    bind_material_system(m);
    
    // Resource types (legacy - use OptiXMaterialType for new code)
    py::enum_<MaterialType>(m, "LegacyMaterialType")
        .value("LAMBERTIAN", MaterialType::LAMBERTIAN)
        .value("PBR", MaterialType::PBR)
        .value("GLASS", MaterialType::GLASS)
        .value("EMISSIVE", MaterialType::EMISSIVE)
        .value("MIRROR", MaterialType::MIRROR);
        
    py::enum_<TextureType>(m, "TextureType")
        .value("RGB", TextureType::RGB)
        .value("RGBA", TextureType::RGBA)
        .value("NORMAL_MAP", TextureType::NORMAL_MAP)
        .value("GRAYSCALE", TextureType::GRAYSCALE)
        .value("HDR", TextureType::HDR);
    
    // ResourceManager
    py::class_<ResourceManager>(m, "ResourceManager")
        .def(py::init<>())
        .def("create_geometry", 
             [](ResourceManager& self, 
                const torch::Tensor& vertices,
                py::object indices = py::none(),
                py::object normals = py::none(),
                py::object tex_coords = py::none(),
                py::object tangents = py::none(),
                py::object bitangents = py::none()) {
                 // Convert Python None to std::nullopt, otherwise convert to tensor
                 std::optional<torch::Tensor> indicesOpt = indices.is_none() ? 
                     std::nullopt : std::optional<torch::Tensor>(indices.cast<torch::Tensor>());
                 std::optional<torch::Tensor> normalsOpt = normals.is_none() ? 
                     std::nullopt : std::optional<torch::Tensor>(normals.cast<torch::Tensor>());
                 std::optional<torch::Tensor> texCoordsOpt = tex_coords.is_none() ? 
                     std::nullopt : std::optional<torch::Tensor>(tex_coords.cast<torch::Tensor>());
                 std::optional<torch::Tensor> tangentsOpt = tangents.is_none() ? 
                     std::nullopt : std::optional<torch::Tensor>(tangents.cast<torch::Tensor>());
                 std::optional<torch::Tensor> bitangentsOpt = bitangents.is_none() ? 
                     std::nullopt : std::optional<torch::Tensor>(bitangents.cast<torch::Tensor>());
                 
                 // Call the C++ method
                 return self.createGeometry(
                     vertices, 
                     indicesOpt, 
                     normalsOpt, 
                     texCoordsOpt, 
                     tangentsOpt, 
                     bitangentsOpt
                 );
             },
             py::arg("vertices"), 
             py::arg("indices") = py::none(),
             py::arg("normals") = py::none(),
             py::arg("tex_coords") = py::none(),
             py::arg("tangents") = py::none(),
             py::arg("bitangents") = py::none())
        .def("create_texture", &ResourceManager::createTexture,
             py::arg("data"), py::arg("type") = TextureType::RGB)
        .def("create_material", &ResourceManager::createMaterial)
        .def("set_material_parameter", [](ResourceManager& self, MaterialHandle handle, const std::string& name, py::object value) {
            // Get the material resource
            MaterialResource* material = self.getMaterial(handle);
            if (!material) {
                throw std::runtime_error("Invalid material handle");
            }
            
            // Special case for texture handles (albedoTexture, normalTexture, etc.)
            if (name.find("Texture") != std::string::npos && py::isinstance<py::int_>(value)) {
                std::cout << name << " = " << py::cast<int>(value) << std::endl;
                // It's a texture handle
                TextureHandle texHandle = value.cast<TextureHandle>();
                
                // Make sure texture exists
                TextureResource* texResource = self.getTexture(texHandle);
                if (!texResource) {
                    throw std::runtime_error("Invalid texture handle");
                }
                
                // Set the texture on the material
                material->setTexture(name, texHandle);
                std::cout<<"texture "<<material->getTexture(name)<<std::endl;
                return true;
            }
            
            // Handle regular parameter types
            if (py::isinstance<py::float_>(value)) {
                // For float parameters
                float floatValue = value.cast<float>();
                material->setParameter(name, MaterialParameter(floatValue));
                return true;
            } 
            if (py::isinstance<py::int_>(value)) {
                // For int parameters
                int intValue = value.cast<int>();
                material->setParameter(name, MaterialParameter(static_cast<float>(intValue)));
                return true;
            }
            if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
                // For list/tuple parameters (assume float3)
                if (py::len(value) != 3) {
                    throw std::runtime_error("List/tuple parameters must have exactly 3 elements for RGB values");
                }

                float3 rgb;
                rgb.x = py::cast<float>(value.attr("__getitem__")(0));
                rgb.y = py::cast<float>(value.attr("__getitem__")(1));
                rgb.z = py::cast<float>(value.attr("__getitem__")(2));

                material->setParameter(name, MaterialParameter(rgb));
                return true;
            }
            if (py::isinstance<torch::Tensor>(value)) {
                // For tensor parameters
                torch::Tensor tensor = value.cast<torch::Tensor>();
                if (tensor.sizes().size() == 0) {
                    // Scalar tensor
                    material->setParameter(name, MaterialParameter(tensor.item<float>()));
                }
                else if (tensor.numel() == 3) {
                    // Vector3 tensor
                    float* rgb = new float[3];
                    rgb[0] = tensor[0].item<float>();
                    rgb[1] = tensor[1].item<float>();
                    rgb[2] = tensor[2].item<float>();
                    material->setParameter(name, MaterialParameter(rgb));
                }
                else {
                    throw std::runtime_error("Tensor must be scalar or have exactly 3 elements");
                }
                return true;
            }

            throw std::runtime_error("Unsupported parameter type");
        }, py::arg("material_handle"), py::arg("name"), py::arg("value"))
        .def("get_available_devices", &ResourceManager::getAvailableDevices);
    
    // Camera parameters
    py::class_<CameraParameters>(m, "CameraParameters")
        .def(py::init<>())
        .def_property("position", 
            // Getter
            [](const CameraParameters& c) -> py::tuple {
                return py::make_tuple(c.position.x, c.position.y, c.position.z);
            },
            // Setter - handles Python list, tuple, or tensor
            [](CameraParameters& c, py::object obj) {
                if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                    if (py::len(obj) != 3) {
                        throw std::runtime_error("Position must be a list/tuple with exactly 3 elements");
                    }
                    // Access list/tuple elements with py::cast<float>(obj.attr("__getitem__")(0))
                    c.position.x = py::cast<float>(obj.attr("__getitem__")(0));
                    c.position.y = py::cast<float>(obj.attr("__getitem__")(1));
                    c.position.z = py::cast<float>(obj.attr("__getitem__")(2));
                } else if (py::isinstance<torch::Tensor>(obj)) {
                    torch::Tensor tensor = obj.cast<torch::Tensor>();
                    if (tensor.numel() != 3) {
                        throw std::runtime_error("Position tensor must have exactly 3 elements");
                    }
                    c.position.x = tensor[0].item<float>();
                    c.position.y = tensor[1].item<float>();
                    c.position.z = tensor[2].item<float>();
                } else {
                    throw std::runtime_error("Position must be a list, tuple, or tensor with 3 elements");
                }
            })
        // Camera basis vectors for direct ray generation
        .def_property("camera_u", 
            // Getter
            [](const CameraParameters& c) -> py::tuple {
                return py::make_tuple(c.u.x, c.u.y, c.u.z);
            },
            // Setter
            [](CameraParameters& c, py::object obj) {
                if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                    if (py::len(obj) != 3) {
                        throw std::runtime_error("camera_u must be a list/tuple with exactly 3 elements");
                    }
                    c.u.x = py::cast<float>(obj.attr("__getitem__")(0));
                    c.u.y = py::cast<float>(obj.attr("__getitem__")(1));
                    c.u.z = py::cast<float>(obj.attr("__getitem__")(2));
                } else if (py::isinstance<torch::Tensor>(obj)) {
                    torch::Tensor tensor = obj.cast<torch::Tensor>();
                    if (tensor.numel() != 3) {
                        throw std::runtime_error("camera_u tensor must have exactly 3 elements");
                    }
                    c.u.x = tensor[0].item<float>();
                    c.u.y = tensor[1].item<float>();
                    c.u.z = tensor[2].item<float>();
                } else {
                    throw std::runtime_error("camera_u must be a list, tuple, or tensor with 3 elements");
                }
            })
        .def_property("camera_v", 
            // Getter
            [](const CameraParameters& c) -> py::tuple {
                return py::make_tuple(c.v.x, c.v.y, c.v.z);
            },
            // Setter
            [](CameraParameters& c, py::object obj) {
                if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                    if (py::len(obj) != 3) {
                        throw std::runtime_error("camera_v must be a list/tuple with exactly 3 elements");
                    }
                    c.v.x = py::cast<float>(obj.attr("__getitem__")(0));
                    c.v.y = py::cast<float>(obj.attr("__getitem__")(1));
                    c.v.z = py::cast<float>(obj.attr("__getitem__")(2));
                } else if (py::isinstance<torch::Tensor>(obj)) {
                    torch::Tensor tensor = obj.cast<torch::Tensor>();
                    if (tensor.numel() != 3) {
                        throw std::runtime_error("camera_v tensor must have exactly 3 elements");
                    }
                    c.v.x = tensor[0].item<float>();
                    c.v.y = tensor[1].item<float>();
                    c.v.z = tensor[2].item<float>();
                } else {
                    throw std::runtime_error("camera_v must be a list, tuple, or tensor with 3 elements");
                }
            })
        .def_property("camera_w", 
            // Getter
            [](const CameraParameters& c) -> py::tuple {
                return py::make_tuple(c.w.x, c.w.y, c.w.z);
            },
            // Setter
            [](CameraParameters& c, py::object obj) {
                if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
                    if (py::len(obj) != 3) {
                        throw std::runtime_error("camera_w must be a list/tuple with exactly 3 elements");
                    }
                    c.w.x = py::cast<float>(obj.attr("__getitem__")(0));
                    c.w.y = py::cast<float>(obj.attr("__getitem__")(1));
                    c.w.z = py::cast<float>(obj.attr("__getitem__")(2));
                } else if (py::isinstance<torch::Tensor>(obj)) {
                    torch::Tensor tensor = obj.cast<torch::Tensor>();
                    if (tensor.numel() != 3) {
                        throw std::runtime_error("camera_w tensor must have exactly 3 elements");
                    }
                    c.w.x = tensor[0].item<float>();
                    c.w.y = tensor[1].item<float>();
                    c.w.z = tensor[2].item<float>();
                } else {
                    throw std::runtime_error("camera_w must be a list, tuple, or tensor with 3 elements");
                }
            });
    
    // Renderer settings
    py::class_<RendererSettings>(m, "RendererSettings")
        .def(py::init<>())
        .def_readwrite("max_bounces", &RendererSettings::maxBounces)
        .def_readwrite("samples_per_pixel", &RendererSettings::samplesPerPixel)
        .def_readwrite("denoise_result", &RendererSettings::denoiseResult);
    
    // GeometryInstance
    py::class_<GeometryInstance, std::shared_ptr<GeometryInstance>>(m, "GeometryInstance")
        .def(py::init<ResourceManager*, GeometryHandle, MaterialHandle, const torch::Tensor&>(),
             py::arg("resource_manager"), py::arg("geometry_handle"), 
             py::arg("material_handle"), py::arg("transform") = torch::eye(4))
        .def("set_transform", (void (GeometryInstance::*)(const torch::Tensor&))&GeometryInstance::setTransform)
        .def("set_transform", (void (GeometryInstance::*)(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&))&GeometryInstance::setTransform,
             py::arg("position"), py::arg("rotation"), py::arg("scale"))
        .def("get_transform", &GeometryInstance::getTransform)
        .def("reset_transform_matrix", &GeometryInstance::resetTransformMatrix)
        .def("set_visible", &GeometryInstance::setVisible)
        .def("is_visible", &GeometryInstance::isVisible)
        .def("set_material", &GeometryInstance::setMaterial);
    
    // Renderer
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<ResourceManager*, int>(),
             py::arg("resource_manager"), py::arg("device_id") = 0)
        .def("initialize", &Renderer::initialize)
        .def("add_instance", &Renderer::addInstance)
        .def("remove_instance", &Renderer::removeInstance)
        .def("clear_instances", &Renderer::clearInstances)
        .def("set_camera", &Renderer::setCamera)
        .def("get_camera", &Renderer::getCamera)
        .def("set_settings", &Renderer::setSettings)
        .def("get_settings", &Renderer::getSettings)
        .def("get_instance_count", &Renderer::getInstanceCount)
        .def("get_traversable", &Renderer::get_traversable)
        .def("debug_force_traversable", &Renderer::debug_force_traversable)
        .def("render", &Renderer::render)
        .def("copy_scene_to", &Renderer::copySceneTo)
        .def("get_last_render_time_ms", &Renderer::getLastRenderTimeMs)
        .def("get_last_copy_time_ms", &Renderer::getLastCopyTimeMs);
}



} // namespace optix_renderer