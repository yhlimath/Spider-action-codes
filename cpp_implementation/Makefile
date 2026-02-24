CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

all: sanity_check main_scaling

sanity_check: sanity_check.o sl3_hecke.o
	$(CXX) $(CXXFLAGS) -o sanity_check sanity_check.o sl3_hecke.o

main_scaling: main_scaling.o sl3_hecke.o
	$(CXX) $(CXXFLAGS) -o main_scaling main_scaling.o sl3_hecke.o

sanity_check.o: sanity_check.cpp sl3_hecke.h
	$(CXX) $(CXXFLAGS) -c sanity_check.cpp

main_scaling.o: main_scaling.cpp sl3_hecke.h
	$(CXX) $(CXXFLAGS) -c main_scaling.cpp

sl3_hecke.o: sl3_hecke.cpp sl3_hecke.h
	$(CXX) $(CXXFLAGS) -c sl3_hecke.cpp

clean:
	rm -f *.o sanity_check main_scaling
