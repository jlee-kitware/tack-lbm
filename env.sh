PV_BUILD=${PV_BUILD:-$HOME/gitsync/paraview_build_cuda}
export PV_BUILD
export PYTHONPATH=$PV_BUILD/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$PV_BUILD/lib:$HOME/gitsync/catalyst_build/lib:$HOME/gitsync/conduit_install/lib:$HOME/gitsync/anari_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
