# Selecting Configs

## CUR on global descriptor 

`wfl.select.by_descriptor.CUR_conf_global()` selects atomic structures based on the global (per-config) descriptors. 


## Furthest point sampling

`wfl.select.by_descriptor.greedy_fps_conf_global()` selects atomic structures using greedy farthest point selection on global (per-config) descriptors


## Flat histogram

`wfl.select.flat_histogram.biased_select_conf()` selects configurations by Boltzmann biased flat histogram on a given quantity (e.g. per-atom enthalpy). The method first construct a histogram of the given quantity. The probability of selecting each atomic configuration is then inversely proportional to the density of a given histogram bin, multiplied by a Boltzmann biasing factor. The biasing factor is exponential in the quantity relative to its lowest value divided by a "temperature" in the same units as the quantity.  


## Convex hull

`wfl.select.convex_hul.select()` finds convex hull in the space of volume, composition and another per-atom property (mainly per-atom energy) and returns configs at the vertices of the convex hull, but only the half that lies below the rest of the points. 


## Simple select

- `wfl.select.simple.by_bool_function()` - applies a boolean filter function to all input configs and returns those that were evaluated as `True`. 
- `wfl.select.simple.by_index()` - returns structures based on the index. 

