#!/usr/bin/env bash

# uninstall-cmake.sh:
# Remove CMake from standard installation locations.
# Tested on Ubuntu.
# Dependencies:
#   - rm

show_help() {
  echo "usage: uninstall-cmake.sh [--verbose|-v]"
}

# Initialize all the option variables.
# This ensures we are not contaminated by variables from the environment.
verbose_mode_enabled=false

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -v|--verbose)
            verbose_mode_enabled=true
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
    
done

rm_options="-rf"

if [ "$verbose_mode_enabled" = true ]; then
  rm_options="${rm_options}v"
fi

rm ${rm_options} /usr/local/share/cmake* \
  /usr/local/lib/cmake \
  /usr/local/doc/cmake* \
  /usr/local/bin/cmake \
  /usr/local/share/aclocal/cmake.m4