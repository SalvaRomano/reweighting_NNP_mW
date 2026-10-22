#!/bin/bash

for dir in N_*; do
    input="$dir/water.lammpstrj"
    output="$dir/water_first_2001.lammpstrj"

    if [[ -f "$input" ]]; then
        echo "Processing $input..."

        awk '
        BEGIN { frame=0 }

        /^ITEM: TIMESTEP$/ {
            frame++
            if (frame > 2001)
                exit
        }

        { print }
        ' "$input" > "$output"

        echo "  -> $output"
    fi
done
