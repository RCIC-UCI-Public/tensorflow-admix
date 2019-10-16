#! /bin/bash
#
# remove 2nd requirement for some cuda libs:
#     libcublas.so.10()(64bit)
#     libcublas.so.10(libcublas.so.10)(64bit)

/usr/lib/rpm/find-requires $* | sed -e '/*\.so\.10\.1(lib/d' 
