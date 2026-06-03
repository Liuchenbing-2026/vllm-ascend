add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        fused_kv_norm_rope_swa_cache_custom_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedKvNormRopeSwaCacheCustom
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_kv_norm_rope_swa_cache_custom ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
