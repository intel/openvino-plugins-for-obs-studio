# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)

add_obs_plugin(NAME obs-backgroundconcealment-ov
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES}
    DEPENDENCIES ovmediapipe)
