#i3cols extr_as_one ~/work/oscNext/level7_v01.04/140000/oscNext_genie_level7_v01.04_pass2.140000.00???0.i3.zst \
#    --outdir ~/work/oscNext/level7_v01.04/140000_i3cols \
#    --keys I3EventHeader \
#           I3MCTree \
#           I3MCWeightDict \
#           MCInIcePrimary \
#           SRTTWOfflinePulsesDC \

i3cols extr_as_one ~/work/oscNext/level7_v01.04/140000/oscNext_genie_level7_v01.04_pass2.140000.000001.i3.zst \
    --outdir ~/work/oscNext/level7_v01.04/140000_i3cols_test \
    --keys I3EventHeader \
           I3MCTree \
           I3MCWeightDict \
           MCInIcePrimary \
           SRTTWOfflinePulsesDC \
           SPEFit2_DC \
           retro_crs_prefit__median__cascade \
           retro_crs_prefit__median__track \
           retro_crs_prefit__median__neutrino \
           retro_crs_prefit__max_llh 
