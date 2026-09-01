// Defaults for the feature macros that gate parts of the GPU data layout.
//
// `register_shaders` sets all of these from the cargo features of the same (lowercase) name, so
// the shader structs match their Rust counterparts. The defaults below are what a standalone
// `slangc` invocation, or a build that never calls `register_shaders`, gets; they mirror slosh's
// own defaults. Slang warns on an undefined identifier in `#if`, so every macro needs one.
//
// Include this from any module that reads one of them:
//
//     #include "slosh/features.h"
//
// This file holds preprocessor directives only and carries no `module` declaration, so it is a
// plain textual include rather than a slang module (the build helpers only compile `*.slang`).

#ifndef SLOSH_FEATURES_H
#define SLOSH_FEATURES_H

// Gates the CPIC parts of Node: the incompatible-momentum lane and the cdf collision field.
// Keep in lockstep with `GpuGridNode` on the Rust side.
#ifndef SLOSH_CPIC
#define SLOSH_CPIC 1
#endif

// Gates the per-node particle linked lists (built in finalize_particles_sort, cleared in reset).
// The bound buffers and the Rust `GridArgs` fields share it so everything stays in sync.
#ifndef SLOSH_NODE_PARTICLE_LISTS
#define SLOSH_NODE_PARTICLE_LISTS 1
#endif

// Gates everything the absorbing perfectly-matched layer adds: the per-axis mass on Node, the
// per-particle mass scale and stiffness damping, and the model itself. Off by default, like the
// cargo feature.
#ifndef SLOSH_PML
#define SLOSH_PML 0
#endif

#endif // SLOSH_FEATURES_H
