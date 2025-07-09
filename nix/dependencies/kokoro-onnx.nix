# kokoro-onnx package with GPU/CPU variants
{ pkgs, prev, system, cudaSupport ? (system == "x86_64-linux"), espeakng_loader, phonemizer_fork }:

prev.buildPythonPackage rec {
  pname = "kokoro_onnx";
  version = "0.4.9-git";
  format = "pyproject";
  
  src = pkgs.fetchFromGitHub {
    owner = "thewh1teagle";
    repo = "kokoro-onnx";
    rev = "main";  # Use main branch
    hash = "sha256-PcGNRT1erpcZi6G/MrTbdpqfz2WqzhldNtu/hAANKYw=";
  };
  
  nativeBuildInputs = with prev; [
    hatchling
    hatch-vcs
  ];

  propagatedBuildInputs = with prev; [
    numpy
    huggingface-hub
    onnxruntime
    colorlog
  ] ++ [ espeakng_loader phonemizer_fork ];
  
  # Add hf-transfer support
  passthru.optional-dependencies.gpu = with prev; [
    hf-transfer
  ];
  
  doCheck = false;
  
  # Disable runtime dependencies check as some deps are optional
  dontCheckRuntimeDeps = true;
  
  meta = with pkgs.lib; {
    description = "ONNX version of Kokoro TTS";
    homepage = "https://github.com/thewh1teagle/kokoro-onnx";
    license = licenses.asl20;
    platforms = platforms.unix;
  };
}