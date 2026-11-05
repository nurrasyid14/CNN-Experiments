add_library(cnn_ingestion STATIC
  ingestion/file_validator.cpp
  ingestion/image_reader.cpp
  ingestion/metadata_reader.cpp)
target_include_directories(cnn_ingestion
  PUBLIC include include/ingestion
  PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cnn_ingestion
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE ${OpenCV_LIBS})

add_executable(cnn_ingest ingestion/main.cpp)
target_link_libraries(cnn_ingest PRIVATE cnn_ingestion)
