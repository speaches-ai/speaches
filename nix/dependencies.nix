# Custom package dependencies for speaches
# All packages are built from GitHub source (not PyPI wheels).
{
  pkgs,
  pyPackages,
  system,
}:
rec {
  espeakng_loader = pyPackages.buildPythonPackage {
    pname = "espeakng_loader";
    version = "0.1.0";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "thewh1teagle";
      repo = "espeakng-loader";
      rev = "0ddc87adf77e5850d7eeb542ac8a87d421b64daa";
      hash = "sha256-nSEQ9rofFl6BTH18L5DzaQ1Ymw5H3d+wSEXUxp4o1DM=";
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
      rev = "2bfb160cfae06709a6d7c3d436293972e0f1d12f";
      hash = "sha256-UlPhijY9UHKcck30C0mQ5CcN3Zi/TzARjpzAsYkUqxc=";
    };
    nativeBuildInputs = with pyPackages; [
      hatchling
      hatch-vcs
    ];
    propagatedBuildInputs =
      with pyPackages;
      [
        numpy
        huggingface-hub
        onnxruntime
        colorlog
      ]
      ++ [
        espeakng_loader
        phonemizer_fork
      ];
    doCheck = false;
  };

  aioice = pyPackages.buildPythonPackage {
    pname = "aioice";
    version = "0.10.2";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "aiortc";
      repo = "aioice";
      tag = "0.10.2";
      hash = "sha256-UEXkTxcpe6mlA2FmMSfDmtcEYE9zwuitpi2Eh188xZc=";
    };
    propagatedBuildInputs = with pyPackages; [
      dnspython
      ifaddr
    ];
    doCheck = false;
  };

  pylibsrtp = pyPackages.buildPythonPackage {
    pname = "pylibsrtp";
    version = "1.0.0";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "aiortc";
      repo = "pylibsrtp";
      tag = "1.0.0";
      hash = "sha256-Q8EyGAJKkq14sqSEMWLB8arKvj/wuALK/XwOZ27F1nQ=";
    };
    nativeBuildInputs = [ pyPackages.cffi ];
    buildInputs = [
      pkgs.srtp
      pkgs.openssl
    ];
    propagatedBuildInputs = [ pyPackages.cffi ];
    doCheck = false;
  };

  aiortc = pyPackages.buildPythonPackage rec {
    pname = "aiortc";
    version = "1.14.0";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "aiortc";
      repo = "aiortc";
      tag = "1.14.0";
      hash = "sha256-ZgxSaiKkJrA5XvUT1zq8kwqB8mOvn46vLWXHyJSsHbM=";
    };
    propagatedBuildInputs =
      with pyPackages;
      [
        pyee
        pyopenssl
        cryptography
        av
        google-crc32c
      ]
      ++ [
        aioice
        pylibsrtp
      ];
    buildInputs = with pkgs; [
      ffmpeg-full
      libvpx
      libopus
      srtp
    ];
    doCheck = false;
  };

  # Override csvw to avoid frictionless -> moto -> sagemaker -> google-pasta
  # dependency chain which is broken on Python 3.14+.
  # nixpkgs' csvw incorrectly includes frictionless as a runtime dep (it is only a test dep).
  csvw = pyPackages.buildPythonPackage {
    pname = "csvw";
    version = "3.7.0";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "cldf";
      repo = "csvw";
      tag = "v3.7.0";
      hash = "sha256-HftvI4xJy/MX0WTIFNyZqNqIJIlHsWhhURpeQ1XqrT0=";
    };
    nativeBuildInputs = [ pyPackages.setuptools ];
    propagatedBuildInputs = with pyPackages; [
      attrs
      isodate
      python-dateutil
      rfc3986
      uritemplate
      babel
      requests
      language-tags
      rdflib
      termcolor
      jsonschema
    ];
    dontCheckRuntimeDeps = true;
    doCheck = false;
  };

  # Override segments to use our csvw (without frictionless)
  segments = pyPackages.buildPythonPackage {
    pname = "segments";
    version = "2.4.0";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "cldf";
      repo = "segments";
      tag = "v2.4.0";
      hash = "sha256-XhJH87Bb9wGNPpPymRjgPYLv2zr4hGAyIAbTMk0uCU0=";
    };
    nativeBuildInputs = [ pyPackages.setuptools ];
    propagatedBuildInputs = with pyPackages; [
      regex
      csvw
    ];
    dontCheckRuntimeDeps = true;
    doCheck = false;
  };

  phonemizer_fork = pyPackages.buildPythonPackage {
    pname = "phonemizer-fork";
    version = "3.3.2";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "thewh1teagle";
      repo = "phonemizer-fork";
      rev = "2d74b9863f48f98557f3605fdb434c928629861d";
      hash = "sha256-0exVEQgi/+L9V0h+K9lUaWICtmILRnb//izGyjOVID0=";
    };
    # dlinfo is a glibc-only API (Linux); on macOS the unconditional
    # `import dlinfo` in phonemizer's espeak backend crashes at import time.
    # Wrap it in try/except so the module loads. The dlinfo fallback in
    # _shared_library_path is never reached because kokoro_onnx passes an
    # absolute library path via espeakng_loader.
    postPatch = pkgs.lib.optionalString (!pkgs.stdenv.isLinux) ''
      ${pyPackages.python.interpreter} -c "
      import pathlib
      p = pathlib.Path('phonemizer/backend/espeak/api.py')
      s = p.read_text()
      s = s.replace('    import dlinfo', '    try:\n        import dlinfo\n    except ImportError:\n        dlinfo = None')
      p.write_text(s)
      "
    '';
    nativeBuildInputs = [ pyPackages.hatchling ];
    propagatedBuildInputs =
      with pyPackages;
      [
        joblib
        segments
        attrs
        typing-extensions
      ]
      ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
        (pyPackages.dlinfo.overridePythonAttrs (old: {
          doCheck = false;
        }))
      ];
    # dlinfo is Linux-only (glibc API); skip the runtime dep check on macOS
    # since the wheel metadata unconditionally declares it
    dontCheckRuntimeDeps = !pkgs.stdenv.isLinux;
    doCheck = false;
  };

  # onnx_asr has a custom hatch build hook that generates .onnx preprocessor models
  # at build time using torch + onnxscript. These are deterministic signal processing
  # graphs (FFT, mel filterbanks, resamplers), not trained weights.
  onnx_asr = pyPackages.buildPythonPackage {
    pname = "onnx_asr";
    version = "0.10.2";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "istupakov";
      repo = "onnx-asr";
      tag = "v0.10.2";
      hash = "sha256-KumdelY9oNMAEBSGVdvbBH6SYi93n2cA/eEqaE8MmIU=";
    };
    env.SETUPTOOLS_SCM_PRETEND_VERSION = "0.10.2";
    nativeBuildInputs = with pyPackages; [
      hatchling
      hatch-vcs
      # Build deps for the custom hatch hook that generates .onnx preprocessor models
      onnx
      onnxscript
      torch
      torchaudio
    ];
    propagatedBuildInputs = with pyPackages; [
      numpy
      onnxruntime
      huggingface-hub
    ];
    doCheck = false;
  };

  # onnx-diarization and its dependencies
  einops = pyPackages.buildPythonPackage {
    pname = "einops";
    version = "0.8.2";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "arogozhnikov";
      repo = "einops";
      tag = "v0.8.2";
      hash = "sha256-d5Vbtkw/MChS2j2IC6j97wfVoKWZT9mU4OeXyEjm6ys=";
    };
    nativeBuildInputs = [ pyPackages.hatchling ];
    doCheck = false;
  };

  kaldi_native_fbank =
    let
      kissfft-src = pkgs.fetchurl {
        url = "https://github.com/mborgerding/kissfft/archive/febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip";
        hash = "sha256-SXED5mQWjr45WAt1etvmFvbPhaFlcq9YHKe8QtCrE/0=";
      };
    in
    pyPackages.buildPythonPackage {
      pname = "kaldi_native_fbank";
      version = "1.22.3";
      format = "setuptools";
      src = pkgs.fetchFromGitHub {
        owner = "csukuangfj";
        repo = "kaldi-native-fbank";
        rev = "v1.22.3";
        hash = "sha256-Wu4wM52T6NoQ1t5/iAyPtkEGnZki5P0jx0eYMFZMb5o=";
      };
      nativeBuildInputs = with pkgs; [
        cmake
        ninja
      ];
      buildInputs = [
        pyPackages.pybind11
      ];
      propagatedBuildInputs = with pyPackages; [ numpy ];
      postPatch = ''
        # Replace FetchContent-based pybind11 download with find_package
        cat > cmake/pybind11.cmake << 'PYBIND_EOF'
        find_package(pybind11 REQUIRED)
        PYBIND_EOF

        # Place pre-fetched kissfft source where cmake/kissfft.cmake expects it
        cp ${kissfft-src} $PWD/kissfft-febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip
      '';
      dontUseCmakeConfigure = true;
      doCheck = false;
    };

  onnx_dl = pyPackages.buildPythonPackage {
    pname = "onnx_dl";
    version = "0.1.0";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "fedirz";
      repo = "onnx-dl";
      rev = "9ef51fc5e9809441a385bcbc6e7927179f8dbdd2";
      hash = "sha256-tWkxIFLhTFhfNsQkkvfXWdPD1f75wxX9sBFWvV6PGX8=";
    };
    nativeBuildInputs = [ pyPackages.uv-build ];
    postPatch = ''
      sed -i 's/requires = \["uv_build[^"]*"\]/requires = ["uv_build"]/' pyproject.toml
    '';
    propagatedBuildInputs = with pyPackages; [ onnxruntime ];
    doCheck = false;
  };

  pyannote_core = pyPackages.buildPythonPackage {
    pname = "pyannote_core";
    version = "6.0.1";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "pyannote";
      repo = "pyannote-core";
      tag = "6.0.1";
      hash = "sha256-r5NkOAzrQGcb6LPi4/DA0uT9R0ELiYuwQkbT1l6R8Mw=";
    };
    nativeBuildInputs = with pyPackages; [
      hatchling
      hatch-vcs
    ];
    propagatedBuildInputs = with pyPackages; [
      numpy
      pandas
      sortedcontainers
    ];
    doCheck = false;
  };

  onnx_diarization = pyPackages.buildPythonPackage {
    pname = "onnx_diarization";
    version = "0.1.0";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "fedirz";
      repo = "onnx-diarization";
      rev = "9662cf34bb160c893caba0ac28e5f9e5f86fe1d7";
      hash = "sha256-3tkgVEqVyOHFYKssJPkSS51eUx9Tw2a2p3jVI2pkQOI=";
    };
    nativeBuildInputs = [ pyPackages.uv-build ];
    postPatch = ''
      sed -i 's/requires = \["uv_build[^"]*"\]/requires = ["uv_build"]/' pyproject.toml
    '';
    propagatedBuildInputs =
      with pyPackages;
      [
        scipy
        scikit-learn
        onnxruntime
      ]
      ++ [
        einops
        kaldi_native_fbank
        onnx_dl
        pyannote_core
      ];
    dontCheckRuntimeDeps = true;
    doCheck = false;
  };

  # Piper TTS packages (Linux only)
  # piper-tts v1.3.0+ (from OHF-Voice/piper1-gpl) embeds espeak-ng directly
  # and no longer depends on the old rhasspy/piper-phonemize library.
  piper_phonemize = null;

  piper_tts =
    let
      isLinux = (system == "x86_64-linux" || system == "aarch64-linux");

      # Override espeak-ng with piper-specific feature flags disabled
      espeak-ng' = pkgs.espeak-ng.override {
        asyncSupport = false;
        klattSupport = false;
        mbrolaSupport = false;
        pcaudiolibSupport = false;
        sonicSupport = false;
        speechPlayerSupport = false;
      };
    in
    if isLinux then
      pyPackages.buildPythonPackage {
        pname = "piper-tts";
        version = "1.4.1";
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "OHF-Voice";
          repo = "piper1-gpl";
          tag = "v1.4.1";
          hash = "sha256-V/ESZMUT1PXxHNN7H2ckTBVOQRRf4c/L2GNtnkXvNpA=";
        };

        patches = [
          ./piper-tts-cmake-system-libs.patch
        ];

        nativeBuildInputs = with pyPackages; [
          cmake
          ninja
          scikit-build
          setuptools
          pkgs.pkg-config
        ];

        dontUseCmakeConfigure = true;

        env.CMAKE_ARGS = builtins.toString [
          (pkgs.lib.cmakeFeature "UCD_STATIC_LIB" "${espeak-ng'.ucd-tools}/libucd.a")
        ];

        buildInputs = [
          espeak-ng'
        ];

        propagatedBuildInputs = with pyPackages; [
          onnxruntime
          pathvalidate
        ];

        postInstall = ''
          ln -s ${espeak-ng'}/share/espeak-ng-data $out/${pyPackages.python.sitePackages}/piper/
        '';

        doCheck = false;
      }
    else
      null;

  mkdocs_render_swagger_plugin = pyPackages.buildPythonPackage {
    pname = "mkdocs-render-swagger-plugin";
    version = "0.1.2";
    pyproject = true;
    src = pkgs.fetchFromGitHub {
      owner = "bharel";
      repo = "mkdocs-render-swagger-plugin";
      tag = "0.1.2";
      hash = "sha256-E8vUPLpw45zDGdi5Oh9jaEcN/h15WG46sXEQ6LSfKc8=";
    };
    build-system = with pyPackages; [ setuptools wheel ];
    dependencies = with pyPackages; [ mkdocs ];
    doCheck = false;
  };

  pytest_antilru = pyPackages.buildPythonPackage {
    pname = "pytest-antilru";
    version = "2.0.0";
    format = "pyproject";
    src = pkgs.fetchFromGitHub {
      owner = "ipwnponies";
      repo = "pytest-antilru";
      tag = "v2.0.0";
      hash = "sha256-k6BzwzDM/q7cUjfDzmtZTGKWq5lFF6yFEoCGp790xY4=";
    };
    nativeBuildInputs = [ pyPackages.poetry-core ];
    propagatedBuildInputs = [ pyPackages.pytest ];
    doCheck = false;
  };

  webvtt_py = pyPackages.buildPythonPackage {
    pname = "webvtt-py";
    version = "0.5.1";
    format = "setuptools";
    src = pkgs.fetchFromGitHub {
      owner = "glut23";
      repo = "webvtt-py";
      tag = "0.5.1";
      hash = "sha256-rsxhZ/O/XAiiQZqdsAfCBg+cdP8Hn56EPbZARkKamdA=";
    };
    doCheck = false;
  };

  # OpenTelemetry instrumentation packages
  # Pinned to 0.55b0 to match the nixpkgs otel ecosystem version
  opentelemetry_instrumentation_asyncio = pyPackages.buildPythonPackage {
    pname = "opentelemetry-instrumentation-asyncio";
    version = "0.55b0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_asyncio";
      version = "0.55b0";
      hash = "sha256-CiS1ehUiFO/XuydUs1Tkd1J0suecWU95d4G43FaYBkQ=";
    };
    nativeBuildInputs = with pyPackages; [
      hatchling
    ];
    propagatedBuildInputs = with pyPackages; [
      opentelemetry-api
      opentelemetry-instrumentation
      opentelemetry-semantic-conventions
      wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
  };

  opentelemetry_instrumentation_httpx = pyPackages.buildPythonPackage {
    pname = "opentelemetry-instrumentation-httpx";
    version = "0.55b0";
    format = "pyproject";
    src = pyPackages.fetchPypi {
      pname = "opentelemetry_instrumentation_httpx";
      version = "0.55b0";
      hash = "sha256-Y3jonsbBiX+qrzAAJ16A3X86AMJ8ymj5NNP/Vi3V8qw=";
    };
    nativeBuildInputs = with pyPackages; [
      hatchling
    ];
    propagatedBuildInputs = with pyPackages; [
      httpx
      opentelemetry-api
      opentelemetry-instrumentation
      opentelemetry-semantic-conventions
      opentelemetry-util-http
      wrapt
    ];
    doCheck = false;
    dontCheckRuntimeDeps = true;
  };
}
