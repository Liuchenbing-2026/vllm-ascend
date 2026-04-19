add_ops_compile_options(
        OP_NAME ComputeSlotMapping
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        compute_slot_mapping_def.cpp
)

target_sources(optiling PRIVATE
        compute_slot_mapping_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        compute_slot_mapping_infershape.cpp
)
