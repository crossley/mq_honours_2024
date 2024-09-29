#!/bin/zsh
#
# table_exp_summary
# table_dbm

declare -A file_mappings=(
    ["../figures/categories.pdf"]="../figures/fig_1.pdf"
    ["../figures/network_fig.pdf"]="../figures/fig_2.pdf"
    ["../figures/func_fits.pdf"]="../figures/fig_3.pdf"
    ["../figures/param_fits.pdf"]="../figures/fig_4.pdf"
    ["../figures/switch_costs_iirb.pdf"]="../figures/fig_5.pdf"
    ["../figures/switch_costs.pdf"]="../figures/fig_6.pdf"
    ["../figures/categories_3.pdf"]="../figures/fig_7.pdf"
    ["../figures/model_all.pdf"]="../figures/fig_8.pdf"
)


# Loop through the file mappings and copy the files
for source_name in "${(@k)file_mappings}"; do
  target_name="${file_mappings[$source_name]}"
  cp "$source_name" "$target_name"
  echo "Copied $source_name to $target_name"
done
