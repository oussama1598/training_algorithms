set(SOURCES Regression.h Regression.cpp)

add_library(regression ${SOURCES})
target_include_directories(regression INTERFACE ..)
target_link_libraries(regression PRIVATE math)
