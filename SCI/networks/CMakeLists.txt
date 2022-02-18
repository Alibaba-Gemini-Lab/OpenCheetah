set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

macro (add_network_OT _name)
  add_executable(${_name}-SCI_OT "main_${_name}.cpp")
  target_link_libraries(${_name}-SCI_OT SCI-OT) 
  target_compile_definitions(${_name}-SCI_OT PUBLIC SCI_OT=1 USE_CHEETAH=0)
endmacro()

macro (add_network_HE _name)
  add_executable(${_name}-SCI_HE "main_${_name}.cpp")
  target_link_libraries(${_name}-SCI_HE SCI-HE) 
  target_compile_definitions(${_name}-SCI_HE PUBLIC SCI-OT=0 USE_CHEETAH=0)
endmacro()

# Networks
add_network_HE(sqnet)
add_network_HE(resnet50)
add_network_HE(densenet121)
