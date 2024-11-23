target_include_directories(fgprs PUBLIC dat)
include_directories(dat)
add_subdirectory(dat)

target_include_directories(fgprs PUBLIC net)
include_directories(net)
add_subdirectory(net)

target_include_directories(fgprs PUBLIC sch)
include_directories(sch)
add_subdirectory(sch)

target_include_directories(fgprs PUBLIC tests)
include_directories(tests)
add_subdirectory(tests)