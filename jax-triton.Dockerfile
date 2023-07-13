
FROM rocm/jax:rocm5.5.0-jax0.4.12-py3.10.0

# Triton depends on PyTorch
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.5

# Fetch the Triton dev bits for ROCm and install
RUN cd ~ && git clone -b FA_triton_benchmark https://github.com/ROCmSoftwarePlatform/triton.git
RUN git clone -b FA_triton_benchmark https://github.com/ROCmSoftwarePlatform/flash-attention.git
RUN pip install pandas
RUN cd ~/triton/python && pip install -v -e .
