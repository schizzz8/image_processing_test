add_executable(imregionalmax imregionalmax.cpp
)

target_link_libraries(imregionalmax 
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)
