// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "nlohmann/json.hpp"

#include "rknn_qwen3_embedding.h"
#include "vector_config.h"
#include "image_utils.h"
#include "time_utils.h"
#include "cnpy.h"


int64_t first_token;
bool first_decode = true;

struct embedding_info
{
    int fd;
    float16 *embedding_data;
    int embedding_dim;
    int vocab_size;
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.0,
    .temperature = 1.0f,
    .repeat_penalty = 1.0f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f};

const char *system_prompt = "";
const char *prompt_prefix = "";
const char *prompt_postfix = "";

/*-------------------------------------------
                Callback Function
-------------------------------------------*/

int output_callback(void *userdata, rknn3_tensor *output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state)
{

    printf("\noutput_callback: state = %d\n", state);
    if (state != RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED)
    {
        return 0;
    }
    else if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED)
    {
        if (first_decode)
        {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }

        float *model_output = (float *)userdata;

        for (int i = 0; i < n_output_tensors; i++)
        {
            printf("output_callback: output[%d]->attr->index = %d\n", i, output_tensors[i].attr->index);
            printf("output_callback: output[%d]->attr->name = %s\n", i, output_tensors[i].attr->name);
            printf("output_callback: output[%d]->mem->size = %lu\n", i, output_tensors[i].mem->size);
            for (int j = 0; j < output_tensors[i].attr->n_elems; j++)
            {
                model_output[j] = fp16_to_fp32(((float16 *)output_tensors[i].mem->virt_addr)[j]);

                if (j < 10)
                {
                    printf("output_callback: output[%d][%d] = %f\n", i, j, model_output[j]);
                }
            }
        }
    }

    return 0;
}

int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        // Get token text
        std::string piece;
        if (result->num_tokens == 1) {
          piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
          piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        // Print token text
        printf("%s", piece.c_str());

        if (first_decode)
        {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}

int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
    int n_tokens = 0;
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

int embed_callback(void *userdata, int32_t *tokens, uint64_t num_tokens, void *embed, uint64_t len)
{
    struct embedding_info *embed_info = (struct embedding_info *)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16))
    {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++)
    {
        memcpy((unsigned char *)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p)
{

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3 || strcmp(argv[1], "--config") != 0)
    {
        printf("Usage: %s --config config.json\n", argv[0]);
        return -1;
    }

    VectorWorkerConfig worker_config;
    if (!load_vector_worker_config(argv[2], "embedding", &worker_config)) return -1;
    const char *model_path = worker_config.model_path.c_str();
    const char *weight_path = worker_config.weight_path.c_str();
    const char *tokenizer_path = worker_config.tokenizer_path.c_str();
    const char *embedding_path = worker_config.embedding_path.c_str();
    const char *device_id = worker_config.device_id.c_str();
    uint32_t core_mask = worker_config.core_mask;

    int ret;
    rknn_perf_metrics_t perf;

    uint32_t npy_shape[1];
    uint32_t npy_ndim = 1;

    // RKNN Context
    rknn_qwen3_embedding_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_qwen3_embedding_context));
    rknn3_session *session = NULL;
    rknn3_input_output_num io_num;
    memset(&io_num, 0, sizeof(rknn3_input_output_num));

    // input/output tensors will be allocated after querying io_num
    rknn3_tensor *inputs = NULL;
    rknn3_tensor *outputs = NULL;

    // LLM Config
    rknn3_llm_config llm_config;
    memset(&llm_config, 0, sizeof(rknn3_llm_config));

    // set output tensors index for output callback
    rknn3_tensor output_tensors[1];
    int n_output_tensors = 1;
    int output_tensors_index[1] = {0};
    float *model_output = nullptr;
    memset(output_tensors, 0, sizeof(output_tensors));

    // Tokenizer
    Tokenizer *tokenizer;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // LLM Input Tensor
    int n_inputs = 1;
    rknn3_llm_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }

    tokenizer->GetVocabInfo(&(vocab_info));
    printf("vocab_info: vocab_size=%d, special_bos_id=[", vocab_info.vocab_size);
    for (int i = 0; i < vocab_info.n_special_bos_id; ++i)
    {
        printf("%d%s", vocab_info.special_bos_id[i], (i + 1 < vocab_info.n_special_bos_id) ? ", " : "");
    }
    printf("], special_eos_id=[");
    for (int i = 0; i < vocab_info.n_special_eos_id; ++i)
    {
        printf("%d%s", vocab_info.special_eos_id[i], (i + 1 < vocab_info.n_special_eos_id) ? ", " : "");
    }
    printf("]\n");

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1)
    {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1)
    {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16 *)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED)
    {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    printf("--> init qwen3 llm model\n");
    ret = init_qwen3_embedding_model(&rknn_app_ctx, model_path, weight_path, device_id, core_mask);
    if (ret != 0)
    {
        printf("init_qwen3_embedding fail! ret=%d model_path=%s weight_path=%s\n", ret, model_path, weight_path);
        goto out;
    }

    // Query input and output information
    ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS)
    {
        printf("rknn3_query io_num fail! ret=%d\n", ret);
        goto out;
    }

    // allocate input/output arrays based on queried counts
    inputs = (rknn3_tensor *)calloc(io_num.n_input, sizeof(rknn3_tensor));
    outputs = (rknn3_tensor *)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (!inputs || !outputs)
    {
        printf("failed to allocate input/output tensor arrays\n");
        goto out;
    }

    // query input tensors info
    printf("input tensors:\n");
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        inputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        inputs[i].attr->index = i;
        ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_INPUT_ATTR, inputs[i].attr, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query error! ret=%d\n", ret);
            goto out;
        }
        dump_tensor_attr(inputs[i].attr);
    }

    // query output tensors info
    printf("output tensors:\n");
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        outputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        outputs[i].attr->index = i;
        ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, outputs[i].attr, sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_query fail! ret=%d\n", ret);
            goto out;
        }
        dump_tensor_attr(outputs[i].attr);
    }

    for (int i = 0; i < n_output_tensors; i++)
    {
        output_tensors[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        // Query output tensor info according to the output tensor index
        output_tensors[i].attr->index = output_tensors_index[i];
        ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, output_tensors[i].attr, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto out;
        }

        output_tensors[i].mem =
            rknn3_create_mem(rknn_app_ctx.rknn_ctx, output_tensors[i].attr->aligned_size, output_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);

        model_output = (float *)malloc(output_tensors[i].attr->n_elems * sizeof(float));
        if (!model_output)
        {
            printf("Failed to allocate memory for model output!\n");
            goto out;
        }
    }

    ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
    if (ret != RKNN3_SUCCESS)
    {
        printf("rknn3_query llm config failed! ret=%d\n", ret);
        goto out;
    }

    // Set LLM parameters
    params.logits_name = "output";
    params.max_context_len = llm_config.max_ctx_len;
    params.sampling_param = SAMPLE_PARAMS; // not used for embedding model
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));

    // LLM Callback
    callback.result_callback = result_callback;
    callback.result_userdata = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback = embed_callback;
    callback.embed_userdata = &embedding_info;
    callback.output_callback = output_callback;
    callback.output_userdata = model_output;
    callback.output_tensors = output_tensors;
    callback.n_output_tensors = n_output_tensors;

    // RKNN Session Init
    session = rknn3_session_init(rknn_app_ctx.rknn_ctx, &params, n_params);
    if (!session)
    {
        printf("Failed to initialize test session\n");
        goto out;
    }

    // Set Callback
    ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0)
    {
        printf("Failed to set callback\n");
        goto out;
    }

    // Set to context
    rknn_app_ctx.rknn_sess = session;

    // LLM Input.  Embedding sessions have keep_history=0, so each request is
    // independent while the model/session remain resident on the NPU.
    tensor.name = "input";
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;

    {
        nlohmann::json ready = {{"ready", true}, {"model", "qwen3-embedding"},
                                {"dimension", output_tensors[0].attr->n_elems}};
        std::cout << ready.dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            nlohmann::json reply;
            try {
                const nlohmann::json request = nlohmann::json::parse(line);
                const std::string id = request.value("id", "");
                std::vector<std::string> texts;
                if (request.contains("input") && request["input"].is_string()) {
                    texts.push_back(request["input"].get<std::string>());
                } else if (request.contains("input") && request["input"].is_array()) {
                    for (const auto& item : request["input"]) texts.push_back(item.get<std::string>());
                } else {
                    throw std::runtime_error("request.input must be a string or string array");
                }
                nlohmann::json vectors = nlohmann::json::array();
                for (const std::string& text : texts) {
                    tensor.prompt = text.c_str();
                    ret = inference_qwen3_embedding(&rknn_app_ctx, tensor, n_inputs, &perf);
                    if (ret != 0) throw std::runtime_error("embedding inference failed: " + std::to_string(ret));
                    nlohmann::json vector = nlohmann::json::array();
                    for (int i = 0; i < output_tensors[0].attr->n_elems; ++i) vector.push_back(model_output[i]);
                    vectors.push_back(std::move(vector));
                }
                reply = {{"id", id}, {"ok", true}, {"data", vectors}};
            } catch (const std::exception& e) {
                reply = {{"ok", false}, {"error", e.what()}};
            }
            std::cout << reply.dump() << std::endl;
        }
        ret = 0;
    }

out:

    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        if (inputs[i].attr)
        {
            free(inputs[i].attr);
        }
        if (inputs[i].mem)
        {
            rknn3_destroy_mem(rknn_app_ctx.rknn_ctx, inputs[i].mem);
        }
    }

    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        if (outputs[i].attr)
        {
            free(outputs[i].attr);
        }
        if (outputs[i].mem)
        {
            rknn3_destroy_mem(rknn_app_ctx.rknn_ctx, outputs[i].mem);
        }
    }

    if (inputs)
    {
        free(inputs);
        inputs = NULL;
    }

    if (outputs)
    {
        free(outputs);
        outputs = NULL;
    }

    for (int i = 0; i < n_output_tensors; i++)
    {
        if (output_tensors[i].attr)
        {
            free(output_tensors[i].attr);
            output_tensors[i].attr = NULL;
        }
        if (output_tensors[i].mem)
        {
            rknn3_destroy_mem(rknn_app_ctx.rknn_ctx, output_tensors[i].mem);
            output_tensors[i].mem = NULL;
        }
    }

    ret = release_qwen3_embedding(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release qwen3 llm fail! ret=%d\n", ret);
    }

    if (embedding_info.fd != -1)
    {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL)
        {
            munmap((void *)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (tokenizer)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    if (model_output)
    {
        free(model_output);
        model_output = NULL;
    }

    return ret;
}
