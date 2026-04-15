add_ops_compile_options(
    OP_NAME NgramMatchExtract
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

target_sources(op_host_aclnn PRIVATE
        ngram_match_extract_def.cpp
)

target_sources(optiling PRIVATE
        ngram_match_extract_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        ngram_match_extract_infershape.cpp
)
