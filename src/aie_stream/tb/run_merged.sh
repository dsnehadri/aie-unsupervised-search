#!/bin/bash
# POST_MERGED: wait for the LN2 test, then x86 bit-identity vs the smvec reference and an aiesim profile
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
until grep -q "PROF_DONE" run_prof_ln2.log 2>/dev/null; do sleep 30; done
export AIE_XPRE="-DPOST_MERGED"
bash run_x86sim_int16_allblocks.sh 20 > x86_merged_run.log 2>&1; rc=$?; echo "X86_DONE rc=$rc" >> x86_merged_run.log
[ $rc -ne 0 ] && exit 1
diff -rq x86simulator_output/data x86_out_smvec_ref > /dev/null && echo "X86_BITIDENTICAL" >> x86_merged_run.log || echo "X86_DIFFERS" >> x86_merged_run.log
/home/snehadri/run_prof_obj.sh /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb merged > run_prof_merged.log 2>&1
echo "PROF_DONE" >> run_prof_merged.log
