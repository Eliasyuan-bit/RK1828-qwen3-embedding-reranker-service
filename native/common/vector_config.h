#pragma once

#include <cstdint>
#include <string>

struct VectorWorkerConfig {
  std::string model_path;
  std::string weight_path;
  std::string tokenizer_path;
  std::string embedding_path;
  std::string device_id;
  uint32_t core_mask = 0;
};

bool load_vector_worker_config(const char* path, const char* worker_name,
                               VectorWorkerConfig* config);
