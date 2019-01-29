CC      	= g++
CFLAGS     	= -std=c++14 -I/home/sakuma/local/include
LDFLAGS    	= -L/home/sakuma/local/lib
LIBS       	= -lboost_system -lboost_filesystem -lboost_program_options -ljpeg -lpng -lpthread
OBJS       	= main.o sub.o
PROGRAM    	= make_multi_synth90k

$(PROGRAM):	$(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) $(LIBS) -o $(PROGRAM)
