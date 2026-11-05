find_package(GTest REQUIRED)
add_executable(ingestion_tests
  test_file_validator.cpp
  test_image_reader.cpp
  test_metadata_reader.cpp
  test_tensor_builder.cpp)
target_include_directories(ingestion_tests
  PRIVATE ${OpenCV_INCLUDE_DIRS} ../../include)
target_link_libraries(ingestion_tests PRIVATE
  cnn_ingestion cnn_preprocessing GTest::gtest GTest::gtest_main ${OpenCV_LIBS})
add_test(NAME ingestion_tests COMMAND ingestion_tests)

if(TARGET cnn_preprocessing_cuda)
  target_sources(ingestion_tests PRIVATE test_cuda_backend.cpp)
  target_link_libraries(ingestion_tests PRIVATE cnn_preprocessing_cuda)
  add_executable(preprocessing_benchmark preprocessing_benchmark.cpp)
  target_include_directories(preprocessing_benchmark PRIVATE ../../include)
  target_link_libraries(preprocessing_benchmark PRIVATE cnn_preprocessing_cuda)
endif()
