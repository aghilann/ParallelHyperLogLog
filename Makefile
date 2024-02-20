NVCC = nvcc

TARGET = HyperLogLog

SOURCES = main.cpp HyperLogLog.cu CudaException.cpp

NVCC_FLAGS = -g -G -Xcompiler -Wall

INCLUDES = -I.

# Specify the libraries (Link with additional libraries if necessary)
LIBS =

# Default make target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Clean up
clean:
	rm -f $(TARGET)
