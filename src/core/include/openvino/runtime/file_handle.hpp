// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <limits>

namespace ov {
#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif

// Same definition is mirrored in openvino/util/file_util.hpp; the guard allows including both.
#ifndef OPENVINO_FILE_REGION_DEFINED
#    define OPENVINO_FILE_REGION_DEFINED
/**
 * @brief A sub-region [offset, offset + size) of the file referenced by a platform-specific file handle.
 *
 * The constructor is intentionally implicit so that callables returning a bare FileHandle remain
 * valid FileHandleProvider implementations and denote the whole file.
 * @ingroup ov_runtime_cpp_api
 */
struct FileRegion {
    /// @brief Platform-specific file handle (int fd on Linux/Unix, HANDLE on Windows).
    FileHandle handle;
    /// @brief Offset in bytes from the beginning of the file where the region starts.
    size_t offset;
    /// @brief Size of the region in bytes. The default maps from offset to the end of the file.
    size_t size;

    constexpr FileRegion(FileHandle handle_,
                         size_t offset_ = 0,
                         size_t size_ = std::numeric_limits<size_t>::max()) noexcept
        : handle(handle_),
          offset(offset_),
          size(size_) {}
};
#endif  // OPENVINO_FILE_REGION_DEFINED

/**
 * @brief Type definition for file handle provider callback (cross-platform).
 * Function that takes no arguments and returns a FileRegion: a platform-specific file handle together with
 * an optional [offset, size) window of the file to be used. A callable returning a bare FileHandle is also
 * accepted and denotes the whole file.
 * The callback implementation must release ownership, caller should close the FileHandle.
 * On Linux/Unix: handle is int (file descriptor)
 * On Windows: handle is void* (HANDLE cast to void*)
 * This is useful for scenarios where file access needs to be controlled externally,
 * such as Android content providers or Windows restricted file access scenarios.
 * @ingroup ov_runtime_cpp_api
 */
using FileHandleProvider = std::function<FileRegion()>;
}  // namespace ov
