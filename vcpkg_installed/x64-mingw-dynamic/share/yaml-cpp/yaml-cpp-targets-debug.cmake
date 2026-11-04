#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yaml-cpp::yaml-cpp" for configuration "Debug"
set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/libyaml-cppd.dll.a"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/libyaml-cppd.dll"
  )

list(APPEND _cmake_import_check_targets yaml-cpp::yaml-cpp )
list(APPEND _cmake_import_check_files_for_yaml-cpp::yaml-cpp "${_IMPORT_PREFIX}/debug/lib/libyaml-cppd.dll.a" "${_IMPORT_PREFIX}/debug/bin/libyaml-cppd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
