/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\<user_name>\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "network.h"
#include "network_data.h"

/* USER CODE BEGIN includes */
// Declare external function
extern void Uart_send(char * str);

static inline char emnist_letter_from_index(uint32_t idx) {
  /* EMNIST Letters trained as 0..25 -> 'A'..'Z' */
  if (idx < 26U) return (char)('A' + idx);
  return '?';
}
/* Quantization parameters derived from ONNX INT8 model */
#define INPUT_SCALE        (0.00392157f)   /* 1/255 */
#define INPUT_ZERO_POINT   (0)
#define OUTPUT_SCALE       (0.06248225f)
#define OUTPUT_ZERO_POINT  (124)
/* Expected input layout: (1,28,28,1) NHWC */
#define IMG_H  (28)
#define IMG_W  (28)
#define IMG_C  (1)
/* RX buffer for 28x28 binary image (0/1). Fill this buffer in your UART ISR */
static volatile uint8_t g_rx_img[IMG_H*IMG_W];
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle network = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_create_and_init");
    return -1;
  }

  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

#if defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
    data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
      ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
    data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
    ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_network_get_error(network),
        "ai_network_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
// Add a static flag to ensure data is processed only once
static int data_processed = 0;

/* Public API to feed 28x28 binary image (0/1 or arbitrary non-zero = 1) */
void AI_SetBinaryImage(const uint8_t* src, uint32_t len)
{
  uint32_t n = (len < (IMG_H*IMG_W)) ? len : (IMG_H*IMG_W);
  for (uint32_t i = 0; i < n; i++) {
    g_rx_img[i] = (src[i] != 0) ? 1 : 0;
  }
  /* allow next acquire_and_process_data to proceed */
  data_processed = 0;
}

int acquire_and_process_data(float* data[])
{
  /* fill the inputs of the c-model
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++ )
  {
      data[idx] = ....
  }
  */

  // Check if data has already been processed
  if (data_processed) {
    return -1;  // Already processed, skip
  }

  /*
   * Quantized input (UINT8): x_fp32 = INPUT_SCALE * (x_u8 - INPUT_ZERO_POINT)
   * Our sender provides 0/1 bitmap; to match x_fp32 in {0,1}, feed x_u8 in {0,255}.
   * Model expects NHWC (1,28,28,1) contiguous layout.
   */
  ai_u8* in_q = (ai_u8*)data_ins[0];
  for (uint32_t y = 0; y < IMG_H; y++) {
    for (uint32_t x = 0; x < IMG_W; x++) {
      uint32_t src_idx = y * IMG_W + x;             /* from g_rx_img (0/1) */
      uint32_t dst_idx = y * (IMG_W * IMG_C) + x;   /* NHWC with C=1 */
      uint8_t v01 = (g_rx_img[src_idx] != 0) ? 255 : 0;  /* 0/255 */
      in_q[dst_idx] = (ai_u8)(v01);
    }
  }

  /* Mark data as ready for processing */
  data_processed = 1;
  return 0;
}

int post_process(float* data[])
{
  char logStr[120];

  /* Output buffer is uint8 quantized */
  ai_u8* out_q = (ai_u8*)data_outs[0];

  /* Compute number of classes safely.
     data_out_1 is ai_i8[] sized AI_NETWORK_OUT_1_SIZE_BYTES.
     So classes = bytes / sizeof(ai_u8) */
  const uint32_t n_classes = (uint32_t)(AI_NETWORK_OUT_1_SIZE_BYTES / sizeof(ai_u8));

  uint32_t argmax = 0;
  float max_logit = -1e30f;

  /* Allocate logits dynamically if you prefer; for 26 it's fine static.
     But keep safe upper bound if you might change to ByClass etc. */
  float logits[128];  /* enough for 26, 47, 62; adjust if needed */
  if (n_classes > (sizeof(logits)/sizeof(logits[0]))) {
    Uart_send("Error: n_classes too large for logits buffer\r\n");
    data_processed = 0;
    return -1;
  }

  /* Dequantize logits and find max */
  float max_val = -1e30f;
  for (uint32_t i = 0; i < n_classes; i++) {
    float l = OUTPUT_SCALE * ((int32_t)out_q[i] - OUTPUT_ZERO_POINT);
    logits[i] = l;
    if (l > max_val) max_val = l;
  }

  /* Softmax (stable) */
  float sum_exp = 0.0f;
  for (uint32_t i = 0; i < n_classes; i++) {
    sum_exp += expf(logits[i] - max_val);
  }
  if (sum_exp <= 0.0f) sum_exp = 1.0f;

  for (uint32_t i = 0; i < n_classes; i++) {
    float prob = expf(logits[i] - max_val) / sum_exp;

    /* argmax */
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      argmax = i;
    }

    /* print index + letter + logit + prob */
    char ch = emnist_letter_from_index(i);
    sprintf(logStr, "%2lu (%c)  logit=%8.5f  prob=%6.3f%%\r\n",
            (unsigned long)i, ch, logits[i], prob * 100.0f);
    Uart_send(logStr);
  }

  /* final prediction as letter */
  sprintf(logStr, "current letter is %c (index %lu)\r\n",
          emnist_letter_from_index(argmax), (unsigned long)argmax);
  Uart_send(logStr);

  data_processed = 0;
  return 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;

  printf("TEMPLATE - run - main loop\r\n");

  if (network) {

      /* 1 - acquire and pre-process input data */
      res = acquire_and_process_data(data_ins);
      /* 2 - process the data - call inference engine */
      if (res == 0)
        res = ai_run();
      /* 3 - post-process the predictions */
      if (res == 0){
        res = post_process(data_outs);
        }
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
