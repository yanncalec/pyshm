#!/bin/sh
# Apply iteratively the same command on all project folders of the database

# if [ "$#" -lt 2 ] || ! [ -d "$2" ]; then
#   echo "Usage: $0 COMMAND DATABASE_DIRECTORY [OPTIONS]" >&2
#   exit 1
# fi

for p in $1/Data/*; do
  if [ -d $p ]; then
    # echo "\n$1 $p/Processed.pkl ${@:3}"
    # $1 $p/Processed.pkl ${@:3}
    echo "./Analysis_of_static_data_VARX.py --component=Seasonal --Ng=24 --Nh=0 -v $p/Processed.pkl"
    ./Analysis_of_static_data_VARX.py --component=Seasonal --Ng=24 --Nh=0 -v $p/Processed.pkl
  fi
done

for p in $1/Outputs/*; do
  if [ -d $p ]; then
    echo "./Analysis_of_static_data_VARX_plot.py --mad -v $p/VARX_[Seasonal]_[Nh=0_Ng=24]/Results.pkl"
    ./Analysis_of_static_data_VARX_plot.py --mad -v $p/VARX_[Seasonal]_[Nh=0_Ng=24]/Results.pkl
  fi
done
