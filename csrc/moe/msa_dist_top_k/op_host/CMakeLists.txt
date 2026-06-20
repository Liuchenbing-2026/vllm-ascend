add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        msa_dist_top_k_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MsaDistTopK
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE msa_dist_top_k ACLNNTYPE aclnn)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        msa_dist_top_k_tiling.cpp
        msa_dist_top_k.cpp
        msa_dist_top_k_split.cpp
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
