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

#ifndef _RKNN_DEMO_QWEN3_RERANKER_UTILS_H_
#define _RKNN_DEMO_QWEN3_RERANKER_UTILS_H_

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

#define MAX_NEW_TOKENS 1

extern const char *system_prompt;
extern const char *prompt_prefix;
extern const char *prompt_postfix;

extern const rknn3_sampling_params SAMPLE_PARAMS;

typedef struct
{
    rknn3_context rknn_ctx;
    rknn3_session *rknn_sess;

} rknn_qwen3_reranker_context;

static void dump_tensor_attr(rknn3_tensor_attr *attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++)
    {
        shape_str += std::to_string(attrs->shape[j]);
        if (j < attrs->n_dims - 1)
        {
            shape_str += ", ";
        }
    }

    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++)
    {
        stride_str += std::to_string(attrs->stride[j]);
        if (j < attrs->n_stride - 1)
        {
            stride_str += ", ";
        }
    }

    printf("    name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

int init_qwen3_reranker_model(rknn_qwen3_reranker_context *llm_ctx, const char *model_path,
                              const char *weight_path, const char *device_id, uint32_t core_mask);

int release_qwen3_reranker(rknn_qwen3_reranker_context *llm_ctx);

int inference_qwen3_reranker(rknn_qwen3_reranker_context *llm_ctx, rknn3_llm_tensor inputs, int n_inputs, rknn_perf_metrics_t *perf);

#endif //_RKNN_DEMO_QWEN3_RERANKER_UTILS_H_
