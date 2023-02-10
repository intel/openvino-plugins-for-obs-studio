# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
find_package(libobs REQUIRED)

#TODO: Figure out a good way to generate a useful plugin version, possibly from git tag / hash.
set( OMZ_OBS_PLUGIN_VERSION 0.4.0 )

set( OMZ_OBS_CONFIGURE_HEADER_IN ${CMAKE_CURRENT_SOURCE_DIR}/plugin-macros.h.in )

function(add_plugins_to_build)
    # check each passed plugin subdirectory
    foreach(dir ${ARGN})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
		    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
                # Include subdirectory to the project.
                add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            endif()
        endif()
    endforeach()
endfunction()

include(CMakeParseArguments)

include_directories(common/include)

#The following macro is defined to be called from the CMakeLists.txt residing within each plugin directory.
# add_obs_plugin(NAME <target name>
#     SOURCES <source files>
#     [HEADERS <header files>]
#     [INCLUDE_DIRECTORIES <include dir>]
#     [OPENCV_VERSION_REQUIRED <X.Y.Z>]
#     [DEPENDENCIES <dependencies>])
macro(add_obs_plugin)
    set(oneValueArgs NAME OPENCV_VERSION_REQUIRED)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(OMZ_OBS_PLUGIN "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
						  
	if(OMZ_OBS_PLUGIN_OPENCV_VERSION_REQUIRED AND OpenCV_VERSION VERSION_LESS OMZ_OBS_PLUGIN_OPENCV_VERSION_REQUIRED)
        message(WARNING "${OMZ_OBS_PLUGIN_NAME} is disabled; required OpenCV version ${OMZ_OBS_PLUGIN_OPENCV_VERSION_REQUIRED}, provided ${OpenCV_VERSION}")
        return()
    endif()
	
	# Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${OMZ_OBS_PLUGIN_SOURCES})
    if(OMZ_OBS_PLUGIN_HEADERS)
        source_group("include" FILES ${OMZ_OBS_PLUGIN_HEADERS})
    endif()
	
	message("generating ${CMAKE_CURRENT_BINARY_DIR}/plugin-macros.generated.h from ${OMZ_OBS_CONFIGURE_HEADER_IN}")
	configure_file(
        ${OMZ_OBS_CONFIGURE_HEADER_IN}
        ${CMAKE_CURRENT_BINARY_DIR}/plugin-macros.generated.h
    )
	
	add_library(${OMZ_OBS_PLUGIN_NAME} MODULE ${OMZ_OBS_PLUGIN_SOURCES} ${OMZ_OBS_PLUGIN_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/plugin-macros.generated.h)
	
	include_directories(
        ${LIBOBS_INCLUDE_DIR}
        ${LIBOBS_INCLUDE_DIRS}
		${CMAKE_CURRENT_BINARY_DIR}
    )
	
	if(WIN32)
        set_target_properties(${OMZ_OBS_PLUGIN_NAME} PROPERTIES COMPILE_PDB_NAME ${OMZ_OBS_PLUGIN_NAME})
    endif()
	
	if(OMZ_OBS_PLUGIN_INCLUDE_DIRECTORIES)
        target_include_directories(${OMZ_OBS_PLUGIN_NAME} PRIVATE ${OMZ_OBS_PLUGIN_INCLUDE_DIRECTORIES})
    endif()
	
	target_link_libraries(${OMZ_OBS_PLUGIN_NAME} PRIVATE OBS::libobs ${OpenCV_LIBRARIES} openvino::runtime
                                                    ${OMZ_OBS_PLUGIN_DEPENDENCIES}  ${ITT_LIBS})
													
	if(UNIX)
        target_link_libraries(${OMZ_OBS_PLUGIN_NAME} PRIVATE pthread)
    endif()
	
	if(WIN32)
	    # Enable Multicore Builds and disable FH4 (to not depend on VCRUNTIME140_1.DLL when building with VS2019)
        if (MSVC)
            add_definitions(/MP /d2FH4- /std:c++latest)
        endif()
		

    endif()
endmacro()

# collect all plugin subdirectories
file(GLOB plugin_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# Build these plugins
add_plugins_to_build(${plugin_dirs})



