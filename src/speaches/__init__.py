import os
import warnings

warnings.filterwarnings("ignore", message="\ntorchcodec is not installed correctly.*", module="pyannote")

DEFAULT_GPU_MEM_LIMIT = 536870912

# Must be set before ctranslate2 is imported (transitively via faster_whisper).
# CT2_CUDA_CACHING_ALLOCATOR_CONFIG format: bin_growth,min_bin,max_bin,max_cached_bytes
_gpu_mem_limit = os.environ.get("GPU_MEM_LIMIT", str(DEFAULT_GPU_MEM_LIMIT))
os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", f"4,3,12,{_gpu_mem_limit}")
