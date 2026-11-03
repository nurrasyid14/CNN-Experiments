# Block 1: Image ingestion

## One-time: vcpkg
```bat
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
setx VCPKG_ROOT C:\vcpkg
```
Open a NEW "x64 Native Tools Command Prompt for VS" afterwards so VCPKG_ROOT and MSVC are both set. Needs CMake >= 3.21.

## Build
```bat
cmake --preset windows-release
cmake --build --preset windows-release
ctest --test-dir build --output-on-failure
```
The first configure compiles OpenCV from source through vcpkg and can take 10-30 minutes. It is cached afterwards.

## Run on SIMPLIcity
```bat
scripts\ingest_simplicity.bat
```
Expected: 1000 ingested, 10 classes x 100, splits train 500 / validation 300 / test 200.

## Options
`cnn_ingest --help`. Label modes: `parent` (class folder), `numeric-stem` (+ `--label-divisor`), `none`.
Bad files never stop the run; they are written to `<manifest>.errors.jsonl` with stage and reason.
