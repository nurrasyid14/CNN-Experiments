find_package(GTest REQUIRED)
add_executable(ingestion_tests
  test_file_validator.cpp
  test_image_reader.cpp
  test_metadata_reader.cpp)
target_include_directories(ingestion_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ingestion_tests PRIVATE
  cnn_ingestion GTest::gtest GTest::gtest_main ${OpenCV_LIBS})
add_test(NAME ingestion_tests COMMAND ingestion_tests)
