//-
// =============================================================================
// Copyright 2022 Autodesk, Inc. All rights reserved.
//
// Use of this software is subject to the terms of the Autodesk license
// agreement provided at the time of installation or download, or which
// otherwise accompanies this software in either electronic or hard copy form.
// =============================================================================
//+

#ifndef PYFROST_EXPORT_H
#define PYFROST_EXPORT_H

#if defined(_WIN32)
#define PYFROST_EXPORT __declspec(dllexport)
#define PYFROST_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define PYFROST_EXPORT __attribute__((visibility("default")))
#define PYFROST_IMPORT __attribute__((visibility("default")))
#else
#error Unsupported platform.
#endif

#if defined(PYFROST_BUILD_NODEDEF_DLL)
#define PYFROST_DECL PYFROST_EXPORT
#else
#define PYFROST_DECL PYFROST_IMPORT
#endif

#endif
