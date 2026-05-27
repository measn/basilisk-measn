include $(BASILISK)/Makefile.defs

EXT_PATH = /home/measn/projects/Basilisk_ext/include

CFLAGS += -O2 \
	-DMOVIE=1 \
	-DUSE_GSL=1 \
	-DUSE_CANTERA=1 \
	-DUSE_OPENSMOKE=0 \
	-DTEST_RESTART=0 \
	-DMTRACE=0 \
	-DTRACE=0 \
	-DBASILISK_SERVER=1 \
	-disable-dimensions \
	-I$(EXT_PATH) \
	-I/home/measn/projects/ecipriano/src \

include Makefile.dynamic
