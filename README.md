# clops
(yet another) OpenCL bindings for GNU Guile

I wrote this set of bindings to Guile in order to learn the OpenCL API.
I only tested it under Windows (using Guile 2.0 provided by MSYS2).
In order to build, it requires [OpenCL headers](https://github.com/KhronosGroup/OpenCL-Headers)
in addition to all the development headers required by Guile.

It should be possible to build it on Linux, but then the Makefile would need to be adjusted.

Running the example in Guile requires presence of the [(grand scheme) glossary](https://github.com/plande/grand-scheme).

I've found [two](https://github.com/te42kyfo/hesp/tree/master/guile-opencl) 
[projects](https://github.com/v01dXYZ/gocl) named "guile-opencl" on the Internet,
so if you're looking for something reliable, you can as well ask there.

In general, the purpose of this project is purely (self)educational: writing
these bindings made me study the documentation thoroughly, and having bindings
available in REPL allows me to experiment with various OpenCL features quickly


I may develop these bindings further to simplify the interfaces,
but at this point in time this is neither a production-quality
library nor a good point to start learning OpenCL.

Mind also, that the event API is not implemented at all.

That being said, if you find anything here useful, enjoy.
