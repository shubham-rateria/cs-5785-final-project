CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -isysroot $(xcrun --show-sdk-path)
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
SOURCE = similarity_matrix.cpp
TARGET = similarity_matrix

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)
