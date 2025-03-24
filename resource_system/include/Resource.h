#pragma once

#include <string>
#include "ResourceTypes.h"

namespace optix_renderer {

// Base class for all resource types
class Resource {
public:
    Resource(ResourceID id, ResourceType type)
        : m_id(id), m_type(type), m_name("") {}
    
    virtual ~Resource() = default;
    
    // Resource identification
    ResourceID getId() const { return m_id; }
    ResourceType getType() const { return m_type; }
    
    // Optional name for the resource
    const std::string& getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

protected:
    ResourceID m_id;
    ResourceType m_type;
    std::string m_name;
};

} // namespace optix_renderer