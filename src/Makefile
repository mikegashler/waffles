SHELL = /bin/bash
export INSTALL_LOCATION_LIB="/usr/local/lib"
export INSTALL_LOCATION_INCLUDE="/usr/local/include"
export INSTALL_LOCATION_BIN="/usr/local/bin"
ifeq ($(UNAME),Darwin)
	export DARWIN_BASE="/usr/X11"
endif
SUBDIRS= _GClasses _wizard _audio _cluster _dimred _learn _plot _recommend _sparse _test _transform
DBG_SUBDIRS= $(SUBDIRS:_%=DBG_%)
OPT_SUBDIRS= $(SUBDIRS:_%=OPT_%)
CLEAN_SUBDIRS= $(SUBDIRS:_%=CLEAN_%)
INSTALL_SUBDIRS= INSTALL_GClasses INSTALL_wizard INSTALL_audio INSTALL_cluster INSTALL_dimred INSTALL_learn INSTALL_plot INSTALL_recommend INSTALL_sparse INSTALL_transform INSTALL_test
UNINSTALL_SUBDIRS= UNINSTALL_GClasses UNINSTALL_wizard UNINSTALL_audio UNINSTALL_cluster UNINSTALL_dimred UNINSTALL_learn UNINSTALL_plot UNINSTALL_recommend UNINSTALL_sparse UNINSTALL_transform UNINSTALL_test

.PHONY: dbg opt install uninstall $(DBG_SUBDIRS) $(OPT_SUBDIRS)

dbg: $(DBG_SUBDIRS)

all: $(DBG_SUBDIRS)

opt: $(OPT_SUBDIRS)

clean: $(CLEAN_SUBDIRS)

configure:
	# There is no need to configure. Just run "make dbg" or "make opt"

install: $(INSTALL_SUBDIRS)

installall: $(INSTALL_SUBDIRS) INSTALL_depends

uninstall: $(UNINSTALL_SUBDIRS)

$(DBG_SUBDIRS): DBG_% :
	$(MAKE) -j 5 -C $* dbg

$(OPT_SUBDIRS): OPT_% :
	$(MAKE) -j 5 -C $* opt

$(CLEAN_SUBDIRS): CLEAN_% :
	$(MAKE) -C $* clean

$(INSTALL_SUBDIRS): INSTALL_% :
	$(MAKE) -j 8 -C $* install

INSTALL_depends:
	$(MAKE) -C depends install

$(UNINSTALL_SUBDIRS): UNINSTALL_% :
	$(MAKE) -C $* uninstall
