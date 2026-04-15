add_ops_compile_options(
    OP_NAME UpdateTokenIdsNgram
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

target_sources(op_host_aclnn PRIVATE
        update_token_ids_ngram_def.cpp
)

target_sources(optiling PRIVATE
        update_token_ids_ngram_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        update_token_ids_ngram_infershape.cpp
)
