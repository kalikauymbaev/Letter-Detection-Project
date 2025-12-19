/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Dec 15 16:39:26 2025
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "33e6737685495dbabc2f2aac0486a6ff"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Dec 15 16:39:26 2025"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 784, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 784, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6272, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3136, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5184, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6272, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 26, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 26, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3328, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 26, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1060, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_1_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _Relu_2_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 258, AI_STATIC)
/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_UINTQ_ZP(0)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023952404037117958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023952404037117958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012415524572134018f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012415524572134018f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05404702574014664f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_GlobalAveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0070475186221301556f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06248224526643753f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06248224526643753f),
    AI_PACK_UINTQ_ZP(152)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016773061826825142f, 0.02326723001897335f, 0.018238157033920288f, 0.013973141089081764f, 0.02288060635328293f, 0.014298276975750923f, 0.007423051167279482f, 0.02345116063952446f, 0.012770788744091988f, 0.01358205359429121f, 0.022062644362449646f, 0.0029710279777646065f, 0.027185970917344093f, 0.0161421075463295f, 0.01627427339553833f, 0.024710822850465775f, 0.014793994836509228f, 0.017574353143572807f, 0.006936623249202967f, 0.021751469001173973f, 0.02571711502969265f, 0.013659831136465073f, 0.013300962746143341f, 0.013734987005591393f, 0.024584198370575905f, 0.015104171819984913f, 0.020682768896222115f, 0.010037130676209927f, 0.015443738549947739f, 0.012132382020354271f, 0.015320319682359695f, 0.02014118619263172f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010760901495814323f, 0.000993165303952992f, 0.0011210486991330981f, 0.0012481256853789091f, 0.0011287047527730465f, 0.0012704675318673253f, 0.0019012567354366183f, 0.0014005096163600683f, 0.001309785759076476f, 0.0012049113865941763f, 0.0015593980206176639f, 0.0016124756075441837f, 0.0012333998456597328f, 0.0013674326473847032f, 0.0006257148925215006f, 0.0017605741741135716f, 0.0011080903932452202f, 0.0011679318267852068f, 0.001712837372906506f, 0.0020890487357974052f, 0.001272063236683607f, 0.001221789512783289f, 0.00134921888820827f, 0.001744959969073534f, 0.0018477782141417265f, 0.0016590855084359646f, 0.0006855973042547703f, 0.001444184104911983f, 0.0013108389684930444f, 0.0013435399159789085f, 0.0018237682525068521f, 0.0016196713550016284f, 0.0016999563667923212f, 0.0012636708561331034f, 0.0025768743362277746f, 0.000867180700879544f, 0.0013149918522685766f, 0.0010267971083521843f, 0.0012107428628951311f, 0.0018958698492497206f, 0.0011543022701516747f, 0.0020175494719296694f, 0.0012296035420149565f, 0.0015381111297756433f, 0.0010615418432280421f, 0.001514891511760652f, 0.0014959042891860008f, 0.0010442911880090833f, 0.0012874933890998363f, 0.0018646679818630219f, 0.0014251201646402478f, 0.0017224275507032871f, 0.0022064652293920517f, 0.0012499046279117465f, 0.0007237523095682263f, 0.001376834698021412f, 0.0012667363043874502f, 0.0009610053966753185f, 0.0013335240073502064f, 0.001713310251943767f, 0.0014461925020441413f, 0.0015288485446944833f, 0.0019137818599119782f, 0.0011188696371391416f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_2_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007106794975697994f, 0.007184814661741257f, 0.006565408781170845f, 0.0071150134317576885f, 0.00598535593599081f, 0.007583890110254288f, 0.006743882317095995f, 0.0071792080998420715f, 0.0071984389796853065f, 0.007540108636021614f, 0.007678274996578693f, 0.008286106400191784f, 0.006656977813690901f, 0.008862176910042763f, 0.00929745752364397f, 0.007004735060036182f, 0.006073656491935253f, 0.007054853718727827f, 0.008113671094179153f, 0.008814729750156403f, 0.008890855126082897f, 0.007982635870575905f, 0.007178617641329765f, 0.005192376207560301f, 0.007703359704464674f, 0.00784747302532196f, 0.007228033617138863f, 0.009482639841735363f, 0.00733548728749156f, 0.0063106706365942955f, 0.00865855347365141f, 0.008873345330357552f, 0.01103421114385128f, 0.011484390124678612f, 0.011457156389951706f, 0.007598384749144316f, 0.007348488550633192f, 0.008103336207568645f, 0.006778500508517027f, 0.00813906081020832f, 0.006949807517230511f, 0.006555848754942417f, 0.007645615842193365f, 0.007099722512066364f, 0.0069247144274413586f, 0.006188249681144953f, 0.007965091615915298f, 0.006344446446746588f, 0.007160259410738945f, 0.007039953488856554f, 0.006872320547699928f, 0.008802984841167927f, 0.007574300281703472f, 0.006588982418179512f, 0.009379800409078598f, 0.008195503614842892f, 0.007647013291716576f, 0.006529358681291342f, 0.00755049055442214f, 0.008550609461963177f, 0.007328021805733442f, 0.008744100108742714f, 0.007638320792466402f, 0.006769232451915741f, 0.006942505948245525f, 0.008413569070398808f, 0.008044739253818989f, 0.006548591889441013f, 0.009938555769622326f, 0.008081845939159393f, 0.006935424637049437f, 0.010879851877689362f, 0.007457911968231201f, 0.007340430282056332f, 0.007157414220273495f, 0.008166305720806122f, 0.008280322887003422f, 0.010802265256643295f, 0.009859105572104454f, 0.008085486479103565f, 0.0057975659146904945f, 0.007180020213127136f, 0.007383617106825113f, 0.0063889892771840096f, 0.006604345515370369f, 0.007008056156337261f, 0.00862649641931057f, 0.006540749222040176f, 0.009054060094058514f, 0.0072733089327812195f, 0.00680844159796834f, 0.005966311786323786f, 0.007767289411276579f, 0.0069304946810007095f, 0.006952608935534954f, 0.008523293770849705f, 0.008833291009068489f, 0.005847859662026167f, 0.0061918203718960285f, 0.007847167551517487f, 0.006886086892336607f, 0.008959216065704823f, 0.007311556953936815f, 0.009330318309366703f, 0.00788850151002407f, 0.007664009463042021f, 0.006215095054358244f, 0.0064790756441652775f, 0.007287829648703337f, 0.007109872996807098f, 0.008295200765132904f, 0.007318299263715744f, 0.008294560015201569f, 0.00813688151538372f, 0.0075065940618515015f, 0.009597750380635262f, 0.014629101380705833f, 0.00934904906898737f, 0.007311652414500713f, 0.006178984418511391f, 0.010217560455203056f, 0.006569050718098879f, 0.007065946701914072f, 0.007474458776414394f, 0.007858000695705414f, 0.005879205651581287f, 0.00727315666154027f, 0.00842730887234211f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 26,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009013857692480087f, 0.008290107361972332f, 0.008015002124011517f, 0.006780604366213083f, 0.007730267941951752f, 0.007157693617045879f, 0.006789277773350477f, 0.008734340779483318f, 0.00837678276002407f, 0.007927902974188328f, 0.006689726375043392f, 0.008928862400352955f, 0.008410212583839893f, 0.009581280872225761f, 0.010426132008433342f, 0.0076582361944019794f, 0.009627201594412327f, 0.011686363257467747f, 0.007009396329522133f, 0.008729360066354275f, 0.007684569340199232f, 0.007627635728567839f, 0.008145050145685673f, 0.00735236844047904f, 0.007544921711087227f, 0.007222442422062159f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023952404037117958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Relu_1_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012415524572134018f),
    AI_PACK_INTQ_ZP(-128)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 28, 28), AI_STRIDE_INIT(4, 1, 1, 1, 28),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_0_conversion_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 28, 28), AI_STRIDE_INIT(4, 1, 1, 1, 28),
  1, &input_0_conversion_output_array, &input_0_conversion_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 14, 14), AI_STRIDE_INIT(4, 1, 1, 32, 448),
  1, &_Relu_output_0_output_array, &_Relu_output_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_pad_before_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &_Relu_1_output_0_pad_before_output_array, &_Relu_1_output_0_pad_before_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 7, 7), AI_STRIDE_INIT(4, 1, 1, 64, 448),
  1, &_Relu_1_output_0_output_array, &_Relu_1_output_0_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_pad_before_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 9, 9), AI_STRIDE_INIT(4, 1, 1, 64, 576),
  1, &_Relu_2_output_0_pad_before_output_array, &_Relu_2_output_0_pad_before_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 7, 7), AI_STRIDE_INIT(4, 1, 1, 128, 896),
  1, &_Relu_2_output_0_output_array, &_Relu_2_output_0_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _GlobalAveragePool_output_0_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_GlobalAveragePool_output_0_output_array, &_GlobalAveragePool_output_0_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 26, 1, 1), AI_STRIDE_INIT(4, 1, 1, 26, 26),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 26, 1, 1), AI_STRIDE_INIT(4, 1, 1, 26, 26),
  1, &logits_QuantizeLinear_Input_0_conversion_output_array, &logits_QuantizeLinear_Input_0_conversion_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_weights, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 1, 32, 96),
  1, &_Relu_output_0_weights_array, &_Relu_output_0_weights_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_Relu_output_0_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_weights, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &_Relu_1_output_0_weights_array, &_Relu_1_output_0_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Relu_1_output_0_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_weights, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &_Relu_2_output_0_weights_array, &_Relu_2_output_0_weights_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_Relu_2_output_0_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 128, 26, 1, 1), AI_STRIDE_INIT(4, 1, 128, 3328, 3328),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 26, 1, 1), AI_STRIDE_INIT(4, 4, 4, 104, 104),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 1060, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1060, 1060),
  1, &_Relu_output_0_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_output_0_scratch1, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 28, 2), AI_STRIDE_INIT(4, 1, 1, 32, 896),
  1, &_Relu_output_0_scratch1_array, &_Relu_output_0_scratch1_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &_Relu_1_output_0_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_1_output_0_scratch1, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 14, 2), AI_STRIDE_INIT(4, 1, 1, 64, 896),
  1, &_Relu_1_output_0_scratch1_array, &_Relu_1_output_0_scratch1_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _Relu_2_output_0_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 9216, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9216, 9216),
  1, &_Relu_2_output_0_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 258, 1, 1), AI_STRIDE_INIT(4, 2, 2, 516, 516),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_0_conversion_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &logits_QuantizeLinear_Input_0_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 32,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _GlobalAveragePool_output_0_layer, 26,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_GlobalAveragePool_output_0_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(7, 7), 
  .pool_stride = AI_SHAPE_2D_INIT(7, 7), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_Relu_2_output_0_weights, &_Relu_2_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_2_output_0_layer, 23,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_3x3_sssa8_ch,
  &_Relu_2_output_0_chain,
  NULL, &_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _Relu_2_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _Relu_2_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _Relu_2_output_0_pad_before_value_data, _Relu_2_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_2_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_2_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Relu_2_output_0_pad_before_layer, 23,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_Relu_2_output_0_pad_before_chain,
  NULL, &_Relu_2_output_0_layer, AI_STATIC, 
  .value = &_Relu_2_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_Relu_1_output_0_weights, &_Relu_1_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_1_output_0_scratch0, &_Relu_1_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_1_output_0_layer, 20,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &_Relu_1_output_0_chain,
  NULL, &_Relu_2_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _Relu_1_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _Relu_1_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _Relu_1_output_0_pad_before_value_data, _Relu_1_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_1_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_1_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Relu_1_output_0_pad_before_layer, 17,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_Relu_1_output_0_pad_before_chain,
  NULL, &_Relu_1_output_0_layer, AI_STATIC, 
  .value = &_Relu_1_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_Relu_output_0_weights, &_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Relu_output_0_scratch0, &_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _Relu_output_0_layer, 14,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &_Relu_output_0_chain,
  NULL, &_Relu_1_output_0_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_0_conversion_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &input_0_conversion_chain,
  NULL, &_Relu_output_0_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 96776, 1, 1),
    96776, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 17600, 1, 1),
    17600, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &input_0_conversion_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 96776, 1, 1),
      96776, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 17600, 1, 1),
      17600, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_QuantizeLinear_Input_0_conversion_output),
  &input_0_conversion_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_network_activations_map[0] + 5004);
    input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5004);
    
    input_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 5004);
    input_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5004);
    
    _Relu_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 5788);
    _Relu_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 5788);
    
    _Relu_output_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 6848);
    _Relu_output_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 6848);
    
    _Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8640);
    _Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8640);
    
    _Relu_1_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 448);
    _Relu_1_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 448);
    
    _Relu_1_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 8640);
    _Relu_1_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 8640);
    
    _Relu_1_output_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 15808);
    _Relu_1_output_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 15808);
    
    _Relu_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Relu_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _Relu_2_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 3136);
    _Relu_2_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3136);
    
    _Relu_2_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 8320);
    _Relu_2_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 8320);
    
    _Relu_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Relu_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    _GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 6272);
    _GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6272);
    
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 516);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 516);
    
    logits_QuantizeLinear_Input_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    logits_QuantizeLinear_Input_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _Relu_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    _Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 288);
    _Relu_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 288);
    
    _Relu_1_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_1_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 416);
    _Relu_1_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 416);
    
    _Relu_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 18848);
    _Relu_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 18848);
    
    _Relu_2_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _Relu_2_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 19104);
    _Relu_2_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 19104);
    
    _Relu_2_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Relu_2_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 92832);
    _Relu_2_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 92832);
    
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_network_weights_map[0] + 93344);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 93344);
    
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_network_weights_map[0] + 96672);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 96672);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 7500238,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 7500238,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

