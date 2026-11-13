#pragma once
#ifndef CUSH_PYTHON_INCLUDE_H
#define CUSH_PYTHON_INCLUDE_H

// Includes Python.h for matplotlib-cpp (Tests and App only; plotting never goes into the DLL).
//
// In a Debug build _DEBUG is defined, and pyconfig.h then links python3XY_d.lib and defines Py_DEBUG, whose ABI
// differs from the release Python that Anaconda and python.org ship. Python.h is therefore included with _DEBUG
// temporarily undefined, as pybind11 does. Standard headers that depend on _DEBUG are included first, so they
// keep seeing the Debug value. Include this header before matplotlibcpp.h in every translation unit.

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_DEBUG)
#define CUSH_RESTORE_DEBUG
#undef _DEBUG
#endif

#include <Python.h>

#if defined(CUSH_RESTORE_DEBUG)
#define _DEBUG
#undef CUSH_RESTORE_DEBUG
#endif

#endif // !CUSH_PYTHON_INCLUDE_H
