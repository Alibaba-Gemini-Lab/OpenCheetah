target_sources(gemini PUBLIC
${CMAKE_CURRENT_LIST_DIR}/shape_inference.cc
${CMAKE_CURRENT_LIST_DIR}/tensor_shape.cc
${CMAKE_CURRENT_LIST_DIR}/tensor_encoder.cc
${CMAKE_CURRENT_LIST_DIR}/hom_conv2d_ss.cc
${CMAKE_CURRENT_LIST_DIR}/hom_fc_ss.cc
${CMAKE_CURRENT_LIST_DIR}/hom_bn_ss.cc
)
