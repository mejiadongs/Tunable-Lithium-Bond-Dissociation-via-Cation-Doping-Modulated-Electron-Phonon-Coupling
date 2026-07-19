# Liquid-Phase Dissociation Models

This submodule contains notebooks and scripts for adsorption modeling, energy analysis, and constrained-structure preparation in liquid or interfacial environments.

## Main Files

- `binding_energy_calculation_full_screening.ipynb`
  Binding-energy workflow for fully screened candidate sets.
- `gibbs_free_energy_change_calculation.ipynb`
  Gibbs free energy change calculations.
- `gibbs_free_energy_postprocessing.ipynb`
  Postprocessing notebook for Gibbs free energy results.
- `surface_li2s_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for Li2S.
- `surface_li2s2_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for Li2S2.
- `surface_li2s3_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for Li2S3.
- `surface_li2s4_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for Li2S4.
- `surface_li2sx_sy_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for mixed sulfur species.
- `surface_lis_li_adsorption_modeling_gap_0.ipynb`
  Surface adsorption model for LiS and Li-related species.
- `freeze_layers_from_db.py`
  Freezes selected layers from database-derived structures.
- `freeze_bottom_fraction_from_db.py`
  Freezes the lower structural fraction from database-derived structures.
- `freeze_all_but_lis_from_db.py`
  Freezes all atoms except LiS-related species from database-derived structures.

## Notes

- The repeated `gap_0` naming suggests these notebooks belong to one consistent setup family.
- If multiple gap settings are added later, consider grouping them under species-specific subdirectories.
