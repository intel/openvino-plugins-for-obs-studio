# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)
file(GLOB HPP_FILES ./*.hpp)

add_obs_plugin(NAME obs-smart-framing-ov
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES} ${HPP_FILES}
    DEPENDENCIES ovmediapipe )
