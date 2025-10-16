cmake_minimum_required(VERSION 3.22) # CMake version check
project(app)                         # Create project "app"
set(CMAKE_CXX_STANDARD 20)           # Enable C++20 standard

add_executable(app "main.cpp" "triangles_withoutbuffer.cpp" "draw_triangles_with_vbo_ibo_instanced.cpp" "draw_triangle_with_vbo_ibo.cpp" "draw_3dcube.cpp" "draw_3dcube_instanced.cpp" "particles_system.cpp" "particles_withobj.cpp" "particles_withobj_texture.cpp" "particles_fixed_uv.cpp" "particles_specialeffect.cpp" "myLearning.cpp" "glm_demo.cpp" "draw_3dcube_instanced_texture.cpp" "draw_3dcube_instanced_texture_img.cpp" "draw_3dcube_instanced_texture_atlas.cpp" "draw_3dcube_instanced_texture_atlas_light.cpp" "draw_3dcube_instanced_texture_atlas_brightlight.cpp" "3dcubeparticle_texture_light.cpp" "3dcubeparticle_web.cpp" "3dcubeparticle_web_gpuinit.cpp" "3dcubeparticle_withobj.cpp")
target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/stb)

set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_BUILD_MONOLITHIC_LIBRARY STATIC)
add_subdirectory("dawn" EXCLUDE_FROM_ALL)

# glm 소스 폴더를 서브프로젝트로 포함
add_subdirectory(glm)

# 앱 타겟에 glm 링크 (둘 중 무엇이든 되는 쪽으로)
if (TARGET glm::glm)
  target_link_libraries(app PRIVATE glm::glm)
elseif (TARGET glm)
  target_link_libraries(app PRIVATE glm)
endif()

if (EMSCRIPTEN)
  # 산출물을 app.html / app.js / app.wasm / app.data 로 만들기
  set_target_properties(app PROPERTIES SUFFIX ".html")

  target_link_libraries(app PRIVATE emdawnwebgpu_cpp webgpu_glfw)

  # ⬇️ 핵심: 파일시스템 + 프리로드 + 메모리 설정
  # - FORCE_FILESYSTEM: stdio/파일 API 사용을 활성화
  # - preload-file: 실제 폴더를 VFS의 /assets 로 매핑
  # - ALLOW_MEMORY_GROWTH/INITIAL_MEMORY: 1M 파티클 초기화 시 OOM 방지
  target_link_options(app PRIVATE
    "SHELL:-sUSE_GLFW=3"
    "SHELL:-sASYNCIFY=1"
    "SHELL:-sFORCE_FILESYSTEM=1"
    # 폴더 전체를 /assets 로 넣는 편이 경로 실수를 줄입니다.
    "SHELL:--preload-file ${CMAKE_SOURCE_DIR}/floatplane@/floatplane"

    # 메모리 (필요시 512MB로 올려보세요: 536870912)
    "SHELL:-sALLOW_MEMORY_GROWTH=1"
    "SHELL:-sINITIAL_MEMORY=268435456"
  )
else()
  target_link_libraries(app PRIVATE webgpu_dawn webgpu_glfw glfw)
endif()


