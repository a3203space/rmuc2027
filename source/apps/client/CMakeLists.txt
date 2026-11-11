set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
if(MSVC)
    add_compile_options(/utf-8)
endif()
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Network Quick WebEngineWidgets WebChannel WebSockets)
find_program(RMUC_BUN_EXECUTABLE bun REQUIRED)
add_custom_target(client_frontend
    COMMAND ${RMUC_BUN_EXECUTABLE} run build
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
add_library(RMUC2027ClientLib STATIC
    src/arena/ArenaConnection.cpp src/arena/ArenaConnection.h
    src/arena/ArenaWindow.cpp src/arena/ArenaWindow.h
    src/arena/ArenaBridge.h src/arena/LocalPage.h)
target_include_directories(RMUC2027ClientLib PUBLIC src)
target_link_libraries(RMUC2027ClientLib PUBLIC Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Quick Qt6::WebEngineWidgets Qt6::WebChannel Qt6::WebSockets)
add_executable(RMUC2027Client src/main.cpp)
target_link_libraries(RMUC2027Client PRIVATE RMUC2027ClientLib)
add_custom_target(client_web ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:RMUC2027Client>/web"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/build/web" "$<TARGET_FILE_DIR:RMUC2027Client>/web")
add_dependencies(client_web RMUC2027Client client_frontend)
if(BUILD_TESTING)
    find_package(Qt6 REQUIRED COMPONENTS Test)
    foreach(kind state connection local_page)
        add_executable(test_arena_${kind} ../../tests/client/test_arena_${kind}.cpp)
        target_link_libraries(test_arena_${kind} PRIVATE RMUC2027ClientLib Qt6::Test)
        add_test(NAME Arena_${kind} COMMAND test_arena_${kind})
        set_tests_properties(Arena_${kind} PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;QT_QUICK_BACKEND=software")
    endforeach()
endif()
