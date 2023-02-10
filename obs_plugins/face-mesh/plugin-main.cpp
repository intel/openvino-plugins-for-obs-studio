// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <obs-module.h>

#include "plugin-macros.generated.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char* obs_module_description(void)
{
    return "Face Mesh filter powered by OpenVINO";
}

extern struct obs_source_info face_mesh_filter_info_ocv;

bool obs_module_load(void)
{
    obs_register_source(&face_mesh_filter_info_ocv);
    blog(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}

void obs_module_unload()
{
    blog(LOG_INFO, "plugin unloaded");
}
