#include "vector_config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

bool load_vector_worker_config(const char* path, const char* worker_name,
                               VectorWorkerConfig* config)
{
  if (!path || !worker_name || !config) return false;
  try {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open configuration file");
    nlohmann::json root;
    stream >> root;
    const nlohmann::json& value = root.at(worker_name);
    config->model_path = value.at("model_path").get<std::string>();
    config->weight_path = value.at("weight_path").get<std::string>();
    config->tokenizer_path = value.at("tokenizer_path").get<std::string>();
    config->embedding_path = value.at("embedding_path").get<std::string>();
    config->device_id = value.at("device_id").get<std::string>();
    const std::string mask = value.at("core_mask").get<std::string>();
    char* end = nullptr;
    config->core_mask = static_cast<uint32_t>(std::strtoul(mask.c_str(), &end, 0));
    if (!end || *end != '\0' || config->core_mask == 0 || config->device_id.empty()) {
      throw std::runtime_error("invalid device_id or core_mask");
    }
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "failed to load %s configuration from %s: %s\n",
                 worker_name, path, error.what());
    return false;
  }
}
