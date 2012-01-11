SHELL = /bin/bash
export INSTALL_LOCATION_INCLUDE="/usr/local/include"
UNAME = $(shell uname -s)
SUBDIRS= _caronhill _chess _hello_console _hello_gui _hello_ml _hello_web _interpolate _journal _manifold _paint _qlearning _raytrace _recommend _ubp

DBG_SUBDIRS= $(SUBDIRS:_%=DBG_%)
OPT_SUBDIRS= $(SUBDIRS:_%=OPT_%)
CLEAN_SUBDIRS= $(SUBDIRS:_%=CLEAN_%)

.PHONY: dbg opt $(DBG_SUBDIRS) $(OPT_SUBDIRS)

opt: check_for_glasses_lib $(OPT_SUBDIRS)

dbg: check_for_glasses_lib $(DBG_SUBDIRS)

ifeq ($(UNAME),Darwin)
check_for_glasses_lib:
else
check_for_glasses_lib:
	@if [ -f /usr/local/lib/libGClasses.a ]; then true; else echo "Error: /usr/local/lib/libGClasses.a not found. This probably means you have not yet installed it. To install it, do: cd ../src; sudo make install"; false; fi
endif

clean: $(CLEAN_SUBDIRS)

configure:
	@echo "There is no need to configure. Just run 'make dbg' or 'make opt'"

install:
	@echo "The demo apps do not need to be installed. Just run 'make opt' or 'make dbg'."

$(DBG_SUBDIRS): DBG_% :
	$(MAKE) -C $*/src dbg

$(OPT_SUBDIRS): OPT_% :
	$(MAKE) -C $*/src opt

$(CLEAN_SUBDIRS): CLEAN_% :
	$(MAKE) -C $*/src clean
