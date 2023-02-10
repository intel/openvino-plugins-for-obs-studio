// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <obs-module.h>

#include "plugin-macros.generated.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char* obs_module_description(void)
{
    return "Smart Framing filter powered by OpenVINO";
}

//extern struct obs_source_info background_removal_filter_info;
extern struct obs_source_info smart_framing_filter_info_ocv;

bool obs_module_load(void)
{
    // obs_register_source(&background_removal_filter_info);
    obs_register_source(&smart_framing_filter_info_ocv);
    blog(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}

void obs_module_unload()
{
    blog(LOG_INFO, "plugin unloaded");
}
