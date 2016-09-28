bin_PROGRAMS = mrcudaexec
mrcudaexec_SOURCES = mrcudaexec.py.template

mrcudaexec$(EXEEXT): mrcudaexec.py.template
	cp $< mrcudaexec$(EXEEXT)
	${SED} -i -- 's/{{ RCUDA_LIBCUDART }}/$(shell echo "${RCUDA_LIBCUDART}" | ${SED} -e 's/\//\\\//g')/g' mrcudaexec$(EXEEXT)
	${SED} -i -- 's/{{ NVIDIA_LIBCUDART }}/$(shell echo "${NVIDIA_LIBCUDART}" | ${SED} -e 's/\//\\\//g')/g' mrcudaexec$(EXEEXT)
	${SED} -i -- 's/{{ MRCUDA_LIBPATH }}/$(shell echo "${libdir}" | ${SED} -e 's/\//\\\//g')/g' mrcudaexec$(EXEEXT)

