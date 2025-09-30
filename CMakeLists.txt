cmake_minimum_required(VERSION 3.22) # CMake version check
project(app)                         # Create project "app"
set(CMAKE_CXX_STANDARD 20)           # Enable C++20 standard

add_executable(app "main.cpp" "triangles_withoutbuffer.cpp" "draw_triangles_with_vbo_ibo_instanced.cpp" "draw_triangle_with_vbo_ibo.cpp" "draw_3dcube.cpp" "draw_3dcube_instanced.cpp" "particles_system.cpp")

set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_BUILD_MONOLITHIC_LIBRARY STATIC)
add_subdirectory("dawn" EXCLUDE_FROM_ALL)

if(EMSCRIPTEN)
  set_target_properties(app PROPERTIES SUFFIX ".html")
  target_link_libraries(app PRIVATE emdawnwebgpu_cpp webgpu_glfw)
  target_link_options(app PRIVATE "-sASYNCIFY=1" "-sUSE_GLFW=3")
else()
  target_link_libraries(app PRIVATE webgpu_dawn webgpu_glfw glfw)
endif()
