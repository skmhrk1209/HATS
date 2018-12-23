g++ -std=c++17 make_dataset.cpp -o make_dataset \
-I/$HOME/local/include -L/$HOME/local/lib \
-lboost_system -lboost_filesystem -lboost_program_options -ljpeg -lpng
