# Custom package dependencies for speaches
{ pkgs, pyPackages, system }:

rec {
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
    nativeBuildInputs = with pyPackages; [ hatchling hatch-vcs ];
    propagatedBuildInputs = with pyPackages; [
      numpy huggingface-hub onnxruntime colorlog 
    ] ++ [ espeakng_loader phonemizer_fork ];
    doCheck = false;
  };

  # Piper packages - Linux only
  piper_phonemize = if system == "x86_64-linux" then
    pyPackages.buildPythonPackage {
      pname = "piper_phonemize";
      version = "1.2.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://github.com/fedirz/piper-phonemize/raw/refs/heads/master/dist/piper_phonemize-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl";
        hash = "sha256-E7/QdVBXIELF5t2NQAdr8kEBqTCvDHoSZUJyFydSJbM=";
      };
      doCheck = false;
    }
  else null;
  
  piper_tts = if pkgs.stdenv.isLinux && piper_phonemize != null then
    pyPackages.buildPythonPackage {
      pname = "piper_tts";
      version = "1.2.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/24/aa/215bced0725cf5b5afe939f86b177c8ddb0d38292a94e85c55b3fcf6d46d/piper_tts-1.2.0-py3-none-any.whl";
        hash = "sha256-80EK6g+AUdihGAUKW5VPrrNvasDabRD8L4oEOo6vJ7U=";
      };
      propagatedBuildInputs = [ piper_phonemize ];
      doCheck = false;
    }
  else null;

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
      pyee pyopenssl cryptography av dnspython ifaddr google-crc32c 
    ] ++ [ aioice pylibsrtp ];
    buildInputs = with pkgs; [ ffmpeg-full libvpx libopus srtp ];
    doCheck = false;
  };

  # OpenTelemetry packages - simplified inline
  opentelemetry_instrumentation_openai = pyPackages.buildPythonPackage {
    pname = "opentelemetry_instrumentation_openai";
    version = "0.37.1";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_openai";
      version = "0.37.1";
      hash = "sha256-SoS5lXJMoE7TvOltnmI2/2M4EGGU74rkwZMFsFEeRpw=";
    };
    nativeBuildInputs = with pyPackages; [ hatchling poetry-core ];
    propagatedBuildInputs = with pyPackages; [
      opentelemetry-api opentelemetry-instrumentation opentelemetry-semantic-conventions
      typing-extensions wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
    pythonImportsCheck = [];
  };

  opentelemetry_instrumentation_openai_v2 = pyPackages.buildPythonPackage {
    pname = "opentelemetry_instrumentation_openai_v2";
    version = "2.1b0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_openai_v2";
      version = "2.1b0";
      hash = "sha256-GEqV+Ewo9Xn7zXixtULTz3XmvR3Jw7jHvkeGoZzbrxM=";
    };
    nativeBuildInputs = with pyPackages; [ hatchling poetry-core ];
    propagatedBuildInputs = with pyPackages; [
      opentelemetry-api opentelemetry-instrumentation opentelemetry-semantic-conventions
      httpx wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
    pythonImportsCheck = [];
  };

  # Other utility packages - simplified inline
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
}