#pragma once

#include <OgreQuaternion.h>
#include <OgreVector3.h>

#include "core/Vec3.h"

namespace pai {

inline Ogre::Vector3 toOgre(const Vec3& v) {
    return {static_cast<Ogre::Real>(v.x), static_cast<Ogre::Real>(v.y), static_cast<Ogre::Real>(v.z)};
}

/// Orientation of a body sitting on the strip: local x across, y up, z forward.
inline Ogre::Quaternion bodyOrientation(const Frame& f) {
    return Ogre::Quaternion(toOgre(f.right), toOgre(f.up), toOgre(f.forward));
}

/// Orientation of a camera looking along f.forward. An Ogre camera looks down
/// its local -z, and f.right points to the viewer's left, so both flip.
inline Ogre::Quaternion cameraOrientation(const Frame& f) {
    return Ogre::Quaternion(toOgre(-f.right), toOgre(f.up), toOgre(-f.forward));
}

}  // namespace pai
