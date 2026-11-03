add_library(cnn_ingestion STATIC
  src/file_validator.cpp
  src/image_reader.cpp
  src/metadata_reader.cpp)
target_include_directories(cnn_ingestion
  PUBLIC include
  PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cnn_ingestion
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE ${OpenCV_LIBS})

add_executable(cnn_ingest src/main.cpp)
target_link_libraries(cnn_ingest PRIVATE cnn_ingestion)
