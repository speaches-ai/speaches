# Core custom package dependencies for speaches
# Optional packages (Piper TTS, OpenTelemetry) are in nix/overlays/
{
  pkgs,
  pyPackages,
  system,
}: rec {
  # Simplified espeakng-loader inline
  espeakng_loader = pyPackages.buildPythonPackage {
    pname = "espeakng_loader";
    version = "0.1.0";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "thewh1teagle";
      repo = "espeakng-loader";
      rev = "main";
      hash = "sha256-YlqlC5/x54y2nz2o4InCXOy802VE2VEDl7SRr3sBcTk=";
    };
    nativeBuildInputs = [ pyPackages.hatchling ];
    propagatedBuildInputs = [ pkgs.espeak-ng ];
    postPatch = ''
      substituteInPlace src/espeakng_loader/__init__.py \
        --replace-fail 'libespeak-ng' '${pkgs.espeak-ng}/lib/libespeak-ng' \
        --replace-fail "Path(__file__).parent / 'espeak-ng-data'" "Path('${pkgs.espeak-ng}/share/espeak-ng-data')"
    '';
    doCheck = false;
  };

  kokoro_onnx = pyPackages.buildPythonPackage rec {
    pname = "kokoro_onnx";
    version = "0.4.9-git";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "thewh1teagle";
      repo = "kokoro-onnx";
      rev = "main";
      hash = "sha256-PcGNRT1erpcZi6G/MrTbdpqfz2WqzhldNtu/hAANKYw=";
    };
    nativeBuildInputs = with pyPackages; [hatchling hatch-vcs];
    propagatedBuildInputs = with pyPackages; [
        numpy
        huggingface-hub
        onnxruntime
        colorlog
      ]
      ++ [espeakng_loader phonemizer_fork];
    doCheck = false;
  };

  # Simplified aiortc-related packages
  aioice = pyPackages.buildPythonPackage {
    pname = "aioice";
    version = "0.9.0";
    format = "setuptools";
    src = pyPackages.fetchPypi {
      pname = "aioice";
      version = "0.9.0";
      hash = "sha256-/CQBscS24ZNy6q6qKP0b2cv2sOQS5IYlKXxTtJXuvR4=";
    };
    propagatedBuildInputs = with pyPackages; [ dnspython ifaddr ];
    doCheck = false;
  };

  pylibsrtp = pyPackages.buildPythonPackage {
    pname = "pylibsrtp";
    version = "0.10.0";
    format = "setuptools";
    src = pyPackages.fetchPypi {
      pname = "pylibsrtp";
      version = "0.10.0";
      hash = "sha256-2AAZEtf1G9BbTqNVF0eTBjF3f9N4ks87/g5UGnQuaZ8=";
    };
    nativeBuildInputs = [ pyPackages.cffi ];
    buildInputs = [ pkgs.srtp pkgs.openssl ];
    propagatedBuildInputs = [ pyPackages.cffi ];
    doCheck = false;
  };

  aiortc = pyPackages.buildPythonPackage rec {
    pname = "aiortc";
    version = "1.9.0";
    format = "setuptools";
    src = pyPackages.fetchPypi {
      pname = "aiortc";
      version = "1.9.0";
      hash = "sha256-A/qnbXbvDlmJrBA4aJiwKTaXVhAiFyMOL81LApxQswM=";
    };
    propagatedBuildInputs = with pyPackages; [
        pyee
        pyopenssl
        cryptography
        av
        dnspython
        ifaddr
        google-crc32c
      ]
      ++ [aioice pylibsrtp];
    buildInputs = with pkgs; [ffmpeg-full libvpx libopus srtp];
    doCheck = false;
  };

  # Utility packages
  pytest_antilru = pyPackages.buildPythonPackage {
    pname = "pytest_antilru";
    version = "2.0.0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "pytest_antilru";
      version = "2.0.0";
      hash = "sha256-SM/zQmSLahzk5TmM8gOWaQXVRrPyvue7VdfLPsh6hfs=";
    };
    nativeBuildInputs = [ pyPackages.poetry-core ];
    propagatedBuildInputs = [ pyPackages.pytest ];
    doCheck = false;
  };

  phonemizer_fork = pyPackages.buildPythonPackage {
    pname = "phonemizer-fork";
    version = "3.3.2";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/64/f1/0dcce21b0ae16a82df4b6583f8f3ad8e55b35f7e98b6bf536a4dd225fa08/phonemizer_fork-3.3.2-py3-none-any.whl";
      hash = "sha256-lzBcdvQYOzgl2uj0wDImX+eMmUbOWMR9S2IWE0kmS3Q=";
    };
    propagatedBuildInputs = with pyPackages; [
      joblib segments attrs
      (dlinfo.overridePythonAttrs (old: { doCheck = false; }))
    ];
    doCheck = false;
  };

  onnx_asr = pyPackages.buildPythonPackage rec {
    pname = "onnx_asr";
    version = "0.7.0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "onnx_asr";
      version = "0.7.0";
      hash = "sha256-iWRsH4ik2MCdYxmvE9xvLD+FkG0Qg7AnqQsjtNOVMUI=";
    };
    nativeBuildInputs = with pyPackages; [ pdm-backend ];
    propagatedBuildInputs = with pyPackages; [ numpy onnxruntime huggingface-hub ];
    doCheck = false;
  };
}