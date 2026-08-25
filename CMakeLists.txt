cmake_minimum_required(VERSION 3.10)
project(rk1828_qwen3_embedding_reranker_service LANGUAGES CXX)

# The SDK remains external so this repository contains only service code.
set(RKNN3_MODEL_ZOO_ROOT "" CACHE PATH "Path to rknn3-model-zoo")
if(NOT RKNN3_MODEL_ZOO_ROOT)
  message(FATAL_ERROR "Set -DRKNN3_MODEL_ZOO_ROOT=/path/to/rknn3-model-zoo")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(${RKNN3_MODEL_ZOO_ROOT}/3rdparty thirdparty.out)
add_subdirectory(${RKNN3_MODEL_ZOO_ROOT}/utils utils.out)

function(add_vector_daemon target source_dir utility_src)
  add_executable(${target}
    ${source_dir}/main.cc
    ${source_dir}/${utility_src}
    native/common/vector_config.cc
  )
  target_include_directories(${target} PRIVATE
    ${source_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/native/common
    ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/rknpu3/include
    ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/tokenizer/include
    ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/json
  )
  target_link_libraries(${target} PRIVATE
    cnpy timeutils fileutils ${LIBRKNNAPI} ${LIBTOKENIZER} dl pthread
  )
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  install(TARGETS ${target} DESTINATION bin)
endfunction()

add_vector_daemon(rk1828_embedding_daemon native/embedding rknn_qwen3_embedding.cc)
add_vector_daemon(rk1828_reranker_daemon native/reranker rknn_qwen3_reranker.cc)
install(FILES ${LIBRKNNAPI} DESTINATION lib)
