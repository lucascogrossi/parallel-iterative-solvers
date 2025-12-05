# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Targets
TARGET = $(BIN_DIR)/parallel_solvers
TARGET_DETAILED = $(BIN_DIR)/detailed_timing

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -I$(INCLUDE_DIR)
CXX_FLAGS = -std=c++17 -O3 -I$(INCLUDE_DIR)

# Source files
CU_SOURCES = $(wildcard $(SRC_DIR)/solvers/*.cu) $(SRC_DIR)/main.cu
CPP_SOURCES = $(wildcard $(SRC_DIR)/solvers/*.cpp) \
              $(wildcard $(SRC_DIR)/utils/*.cpp) \
              $(wildcard $(SRC_DIR)/benchmark/*.cpp)

# Object files
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SOURCES))

ALL_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)/solvers
	@mkdir -p $(BUILD_DIR)/utils
	@mkdir -p $(BUILD_DIR)/benchmark
	@mkdir -p $(BIN_DIR)
	@mkdir -p results

# Link
$(TARGET): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build complete: $(TARGET)"

# Compile CUDA files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build detailed timing version
detailed: directories $(TARGET_DETAILED)

$(TARGET_DETAILED): $(BUILD_DIR)/main_detailed.o $(filter-out $(BUILD_DIR)/main.o, $(ALL_OBJECTS))
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build complete: $(TARGET_DETAILED)"

$(BUILD_DIR)/main_detailed.o: $(SRC_DIR)/main_detailed.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run detailed timing version
run-detailed: $(TARGET_DETAILED)
	./$(TARGET_DETAILED)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean results
clean-results:
	rm -f results/*.csv
	@echo "Results cleaned"

# Full clean
clean-all: clean clean-results

# Phony targets
.PHONY: all detailed directories run run-detailed clean clean-results clean-all info

# Show configuration
info:
	@echo "Project: Parallel Iterative Solvers"
	@echo "NVCC: $(NVCC)"
	@echo "CXX: $(CXX)"
	@echo "Target: $(TARGET)"
	@echo "CUDA sources: $(CU_SOURCES)"
	@echo "C++ sources: $(CPP_SOURCES)"
