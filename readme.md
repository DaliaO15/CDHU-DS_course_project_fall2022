# Face extraction.

Use the following commands to install dlib with CUDA support:
!git clone https://github.com/davisking/dlib.git
!cd dlib
!mkdir build
!cd build\n
!cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
!cmake --build .
!cd ..
!python setup.py install --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1

Note: Make sure to install Visula studio (For Windows), cmake and CUDA drivers before running the above
