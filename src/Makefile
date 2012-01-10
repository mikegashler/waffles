SHELL = /bin/bash
export INSTALL_LOCATION_LIB="/usr/local/lib"
export INSTALL_LOCATION_INCLUDE="/usr/local/include"
export INSTALL_LOCATION_BIN="/usr/local/bin"
SUBDIRS= _GClasses _wizard _audio _cluster _dimred _generate _learn _plot _recommend _sparse _test _transform
DBG_SUBDIRS= $(SUBDIRS:_%=DBG_%)
OPT_SUBDIRS= $(SUBDIRS:_%=OPT_%)
CLEAN_SUBDIRS= $(SUBDIRS:_%=CLEAN_%)
INSTALL_SUBDIRS= INSTALL_GClasses INSTALL_wizard INSTALL_audio INSTALL_cluster INSTALL_dimred INSTALL_generate INSTALL_learn INSTALL_plot INSTALL_recommend INSTALL_sparse INSTALL_transform INSTALL_test
UNINSTALL_SUBDIRS= UNINSTALL_GClasses UNINSTALL_wizard UNINSTALL_audio UNINSTALL_cluster UNINSTALL_dimred UNINSTALL_generate UNINSTALL_learn UNINSTALL_plot UNINSTALL_recommend UNINSTALL_sparse UNINSTALL_transform UNINSTALL_test
INSTALL_UNIX_LIB_SUBDIRS= INSTALL_UNIX_LIB_GClasses

.PHONY: dbg opt install uninstall $(DBG_SUBDIRS) $(OPT_SUBDIRS)

dbg: $(DBG_SUBDIRS)

opt: $(OPT_SUBDIRS)

yo:
	$(MAKE) -C GClasses yo

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
