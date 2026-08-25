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
#include <string>
#include <iostream>

#include "nlohmann/json.hpp"

#include "rknn_qwen3_reranker.h"
#include "vector_config.h"
#include "image_utils.h"
#include "time_utils.h"

int64_t first_token;
bool first_decode = true;

struct reranker_info
{
    int fd;
    float16 *reranker_data;
    int reranker_dim;
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

// Default instruction for reranker
static const char *DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query";

// Qwen3-Reranker chat template prefix and suffix
static const char *RERANKER_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
static const char *RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n";

/*-------------------------------------------
                Helper Functions
-------------------------------------------*/

static std::string format_reranker_prompt(const char *instruction, const char *query, const char *doc)
{
    const char *inst = instruction ? instruction : DEFAULT_INSTRUCTION;
    return std::string(RERANKER_PREFIX) +
           "<Instruct>: " + inst +
           "\n<Query>: " + query +
           "\n<Document>: " + doc +
           "\n" + RERANKER_SUFFIX;
}

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
    struct reranker_info *ranker_info = (struct reranker_info *)userdata;

    if (len != num_tokens * ranker_info->reranker_dim * sizeof(float16))
    {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++)
    {
        memcpy((unsigned char *)embed + n * ranker_info->reranker_dim * sizeof(float16), ranker_info->reranker_data + tokens[n] * ranker_info->reranker_dim,
               ranker_info->reranker_dim * sizeof(float16));
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
    if (!load_vector_worker_config(argv[2], "reranker", &worker_config)) return -1;
    const char *model_path = worker_config.model_path.c_str();
    const char *weight_path = worker_config.weight_path.c_str();
    const char *tokenizer_path = worker_config.tokenizer_path.c_str();
    const char *embedding_path = worker_config.embedding_path.c_str();
    const char *device_id = worker_config.device_id.c_str();
    uint32_t core_mask = worker_config.core_mask;

    int ret;
    rknn_perf_metrics_t perf;
    float reranker_score = 0;
    std::string formatted_prompt;
    // RKNN Context
    rknn_qwen3_reranker_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_qwen3_reranker_context));
    rknn3_session *session = NULL;
    rknn3_input_output_num io_num;
    memset(&io_num, 0, sizeof(rknn3_input_output_num));

    // for qwen3 reranker model, the number of input and output tensors is 9 and 1 respectively
    rknn3_tensor inputs[9];
    memset(inputs, 0, sizeof(inputs));
    rknn3_tensor outputs[1];
    memset(outputs, 0, sizeof(outputs));

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
    memset(&vocab_info, 0, sizeof(VocabInfo));

    // reranker
    struct reranker_info reranker_info;
    struct stat emb_st;
    memset(&reranker_info, 0x00, sizeof(reranker_info));

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

    // Read reranker
    reranker_info.fd = open(embedding_path, O_RDONLY);
    if (reranker_info.fd == -1)
    {
        printf("Failed to open reranker file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(reranker_info.fd, &emb_st) == -1)
    {
        printf("Failed to get reranker file size\n");
        goto out;
    }

    reranker_info.reranker_data = (float16 *)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, reranker_info.fd, 0);
    if (reranker_info.reranker_data == MAP_FAILED)
    {
        printf("Failed to mmap reranker file\n");
        goto out;
    }

    reranker_info.vocab_size = vocab_info.vocab_size;
    reranker_info.reranker_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    printf("--> init qwen3 llm model\n");
    ret = init_qwen3_reranker_model(&rknn_app_ctx, model_path, weight_path, device_id, core_mask);
    if (ret != 0)
    {
        printf("init_qwen3_reranker fail! ret=%d model_path=%s weight_path=%s\n", ret, model_path, weight_path);
        goto out;
    }

    // Query input and output information
    ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS)
    {
        printf("rknn3_query io_num fail! ret=%d\n", ret);
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
            printf("rknn3_query fail! ret=%d\n", ret);
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
        printf("rknn3_query llm config failed! ret=%d", ret);
        goto out;
    }

    // Set LLM parameters
    params.logits_name = "output";
    params.max_context_len = llm_config.max_ctx_len;
    params.sampling_param = SAMPLE_PARAMS; // not used for reranker model
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
    callback.embed_userdata = &reranker_info;
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

    // LLM Input
    tensor.name = "input";
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;

    {
        std::cout << nlohmann::json({{"ready", true}, {"model", "qwen3-reranker"}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            nlohmann::json reply;
            try {
                const nlohmann::json request = nlohmann::json::parse(line);
                const std::string id = request.value("id", "");
                const std::string request_query = request.at("query").get<std::string>();
                const std::string instruction = request.value("instruction", std::string(DEFAULT_INSTRUCTION));
                const auto& documents = request.at("documents");
                if (!documents.is_array()) throw std::runtime_error("request.documents must be an array");
                nlohmann::json scores = nlohmann::json::array();
                for (const auto& item : documents) {
                    const std::string text = item.is_string() ? item.get<std::string>() : item.at("text").get<std::string>();
                    formatted_prompt = format_reranker_prompt(instruction.c_str(), request_query.c_str(), text.c_str());
                    tensor.prompt = formatted_prompt.c_str();
                    ret = inference_qwen3_reranker(&rknn_app_ctx, tensor, n_inputs, &perf);
                    if (ret != 0) throw std::runtime_error("reranker inference failed: " + std::to_string(ret));
                    scores.push_back(model_output[0]); // raw relevance logit; preserve ranking fidelity
                }
                reply = {{"id", id}, {"ok", true}, {"scores", scores}};
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

    ret = release_qwen3_reranker(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release qwen3 llm fail! ret=%d\n", ret);
    }

    if (reranker_info.fd != -1)
    {
        if (reranker_info.reranker_data != MAP_FAILED && reranker_info.reranker_data != NULL)
        {
            munmap((void *)reranker_info.reranker_data, emb_st.st_size);
            reranker_info.reranker_data = NULL;
        }
        close(reranker_info.fd);
        reranker_info.fd = -1;
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
