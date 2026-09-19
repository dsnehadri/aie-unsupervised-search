#!/bin/bash
# x86 bit-identity (all blocks + embed) then aiesim profile of the object block
cd /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb
cp -r x86simulator_output/data x86_out_smvec_ref 2>/dev/null
bash run_x86sim_int16_allblocks.sh 20 > x86_ln2_run.log 2>&1; rc=$?; echo "X86_DONE rc=$rc" >> x86_ln2_run.log
[ $rc -ne 0 ] && exit 1
diff -rq x86simulator_output/data x86_out_smvec_ref > /dev/null && echo "X86_BITIDENTICAL" >> x86_ln2_run.log || echo "X86_DIFFERS" >> x86_ln2_run.log
./run_x86_embed.sh > x86_embed_ln2.log 2>&1; cp x86simulator_output_embed/data/embed_x_out.txt embed_x_out_ln2.txt
/home/snehadri/run_prof_obj.sh /home/snehadri/repos/aie-unsupervised-search/src/aie_stream/tb ln2 > run_prof_ln2.log 2>&1
echo "PROF_DONE" >> run_prof_ln2.log
