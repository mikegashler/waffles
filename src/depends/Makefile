SHELL = /bin/bash
export INSTALL_LOCATION_LIB="/usr/local/lib"
export INSTALL_LOCATION_INCLUDE="/usr/local/include"
export INSTALL_LOCATION_BIN="/usr/local/bin"
ifeq ($(UNAME),Darwin)
	export DARWIN_BASE="/usr/X11"
endif
SUBDIRS= _generate _security
DBG_SUBDIRS= $(SUBDIRS:_%=DBG_%)
OPT_SUBDIRS= $(SUBDIRS:_%=OPT_%)
CLEAN_SUBDIRS= $(SUBDIRS:_%=CLEAN_%)
INSTALL_SUBDIRS= INSTALL_generate INSTALL_security
UNINSTALL_SUBDIRS= UNINSTALL_generate UNINSTALL_security

.PHONY: dbg opt install uninstall $(DBG_SUBDIRS) $(OPT_SUBDIRS)

dbg: $(DBG_SUBDIRS)

opt: $(OPT_SUBDIRS)

clean: $(CLEAN_SUBDIRS)

configure:
	# There is no need to configure. Just run "make dbg" or "make opt"

install: $(INSTALL_SUBDIRS)

uninstall: $(UNINSTALL_SUBDIRS)

$(DBG_SUBDIRS): DBG_% :
	$(MAKE) -C $* dbg

$(OPT_SUBDIRS): OPT_% :
	$(MAKE) -C $* opt

$(CLEAN_SUBDIRS): CLEAN_% :
	$(MAKE) -C $* clean

$(INSTALL_SUBDIRS): INSTALL_% :
	$(MAKE) -C $* install

$(UNINSTALL_SUBDIRS): UNINSTALL_% :
	$(MAKE) -C $* uninstall
