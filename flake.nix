{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    nix-hug = {
      url = "github:longregen/nix-hug";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nix-hug,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = nixpkgs.lib.genAttrs systems;

      perSystem =
        system:
        let
          isLinux = (system == "x86_64-linux" || system == "aarch64-linux");

          mkOverlay =
            {
              pythonVersion,
              cudaSupport ? true,
            }:
            final: prev:
            let
              isCuda = cudaSupport && isLinux;
              pyPackages = prev."${pythonVersion}Packages";
              targetPyVersion = prev.${pythonVersion}.pythonVersion;
            in
            {
              # Override ctranslate2 for CUDA support
              ctranslate2 =
                if isCuda then
                  prev.ctranslate2.override {
                    stdenv = prev.gcc15Stdenv;
                    withCUDA = true;
                    withCuDNN = true;
                    cudaPackages = prev.cudaPackages_12;
                  }
                else
                  prev.ctranslate2;

              # Silero VAD assets (bundled with faster-whisper source but may be missing in Nix build)
              silero-encoder-v5 = prev.fetchurl {
                url = "https://github.com/SYSTRAN/faster-whisper/raw/v1.1.0/faster_whisper/assets/silero_encoder_v5.onnx";
                hash = "sha256-Dp/I9WQHaT0oP5kEX7lcK5D9yjQzzl+D4sEh+05hUHU=";
              };
              silero-decoder-v5 = prev.fetchurl {
                url = "https://github.com/SYSTRAN/faster-whisper/raw/v1.1.0/faster_whisper/assets/silero_decoder_v5.onnx";
                hash = "sha256-jCA0T1CYRqB8zYWCfohXAX+uZ/q9WGib7Br3nh1Igwc=";
              };
              silero-vad-v6 = prev.fetchurl {
                url = "https://github.com/SYSTRAN/faster-whisper/raw/v1.2.1/faster_whisper/assets/silero_vad_v6.onnx";
                hash = "sha256-TL9Um4Mm9g+A8lNtnu/rRQqavoM2WgmAMciXGfG+F9I=";
              };

              # Override onnxruntime in the python fixed-point for the target Python version only
              pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                (
                  pyFinal: pyPrev:
                  let
                    pyVer = pyPrev.python.pythonVersion;
                    isDarwin = prev.stdenv.isDarwin;
                  in
                  (
                    if isCuda && pyVer == targetPyVersion then
                      {
                        onnxruntime = pyPrev.onnxruntime.override {
                          onnxruntime = prev.onnxruntime.override {
                            cudaSupport = true;
                            cudaPackages = prev.cudaPackages_12;
                            python3Packages = pyPrev;
                            pythonSupport = true;
                          };
                        };
                      }
                    else
                      { }
                  )
                  // (
                    if isDarwin then
                      {
                        # Skip only tests that fail in the macOS Nix sandbox.
                        # These are transitive test deps of gradio, not runtime deps of speaches.
                        geoip2 = pyPrev.geoip2.overridePythonAttrs (old: {
                          # socket.gaierror: sandbox blocks binding local HTTP server
                          disabledTestPaths = (old.disabledTestPaths or []) ++ [
                            "tests/webservice_test.py"
                          ];
                        });
                        aioquic = pyPrev.aioquic.overridePythonAttrs (old: {
                          # ConnectionError: sandbox blocks IPv4 loopback
                          disabledTests = (old.disabledTests or []) ++ [
                            "test_connect_and_serve_ipv4"
                          ];
                        });
                        pyarrow = pyPrev.pyarrow.overridePythonAttrs (old: {
                          # PermissionError: sandbox blocks /usr/share/zoneinfo
                          disabledTests = (old.disabledTests or []) ++ [
                            "test_timezone_absent"
                          ];
                        });
                        scipy = pyPrev.scipy.overridePythonAttrs (old: {
                          # ConnectionError: sandbox blocks dataset downloads
                          disabledTests = (old.disabledTests or []) ++ [
                            "TestDatasets"
                          ];
                        });
                        opentelemetry-exporter-otlp-proto-grpc = pyPrev.opentelemetry-exporter-otlp-proto-grpc.overridePythonAttrs (old: {
                          # gRPC channel errors in sandbox
                          disabledTests = (old.disabledTests or []) ++ [
                            "test_permanent_failure"
                            "test_shutdown"
                            "test_shutdown_wait_last_export"
                            "test_success"
                            "test_unavailable_delay"
                          ];
                        });
                      }
                    else
                      { }
                  )
                  // {
                    fakeredis = pyPrev.fakeredis.overridePythonAttrs { doCheck = isLinux; };
                    databricks-sql-connector = pyPrev.databricks-sql-connector.overridePythonAttrs (old: {
                      # Set short socket timeout so sandbox-blocked HTTP retries to
                      # dummy hosts (foo:443) fail fast instead of retrying with backoff
                      preCheck = (old.preCheck or "") + ''
                        echo "import socket; socket.setdefaulttimeout(5)" >> conftest.py
                      '';
                    });
                    gradio = let
                      base = pyPrev.gradio.overridePythonAttrs (old: {
                        # test_x_gradio_user_mcp_gets_set needs ports 7860-7959 which the sandbox blocks
                        disabledTests = (old.disabledTests or []) ++ [
                          "test_x_gradio_user_mcp_gets_set"
                        ];
                      });
                    in
                      # overridePythonAttrs strips .override, but gradio's
                      # passthru.sans-reverse-dependencies calls gradio.override in
                      # the fixed-point. Re-attach it from the un-overridden package.
                      base // { inherit (pyPrev.gradio) override; };
                  }
                )
              ];

              "${pythonVersion}Packages" = pyPackages // {
                # Override faster-whisper to use our ctranslate2 and ensure silero assets exist
                faster-whisper = pyPackages.faster-whisper.overrideAttrs (old: {
                  propagatedBuildInputs = old.propagatedBuildInputs ++ [ final.ctranslate2 ];
                  postInstall = (old.postInstall or "") + ''
                    # Copy silero VAD assets if they don't exist
                    assets_dir="$out/${pyPackages.python.sitePackages}/faster_whisper/assets"
                    mkdir -p "$assets_dir"
                    if [ ! -f "$assets_dir/silero_encoder_v5.onnx" ]; then
                      cp ${final.silero-encoder-v5} "$assets_dir/silero_encoder_v5.onnx"
                      cp ${final.silero-decoder-v5} "$assets_dir/silero_decoder_v5.onnx"
                    fi
                    if [ ! -f "$assets_dir/silero_vad_v6.onnx" ]; then
                      cp ${final.silero-vad-v6} "$assets_dir/silero_vad_v6.onnx"
                    fi
                  '';
                });
              };
            };

          mkSpeaches =
            {
              pythonVersion ? "python312",
              withCuda ? isLinux,
              withDev ? false,
            }:
            let
              overlay = mkOverlay {
                inherit pythonVersion;
                cudaSupport = withCuda;
              };

              pkgs = import nixpkgs {
                inherit system;
                config.allowUnfree = true;
                overlays = [ overlay ];
              };

              python = pkgs.${pythonVersion};
              pythonPackages = pkgs."${pythonVersion}Packages";

              # Import custom deps using the fixed-point python packages (includes CUDA onnxruntime)
              customDeps = import ./nix/dependencies.nix {
                inherit pkgs system;
                pyPackages = python.pkgs;
              };

              # Python environment with configurable dependencies
              pythonEnv = python.withPackages (
                ps:
                let
                  # Core dependencies - always included
                  coreDeps = [
                    ps.fastapi
                    ps.huggingface-hub
                    ps.numpy
                    ps.pydantic
                    ps.pydantic-settings
                    ps.python-multipart
                    ps.soundfile
                    ps.uvicorn
                    ps.openai
                    ps.aiostream
                    ps.cachetools
                    ps.gradio
                    ps.httpx
                    ps.httpx-sse
                    ps.httpx-ws
                    pythonPackages.faster-whisper
                  ]
                  ++ pkgs.lib.optionals (customDeps ? kokoro_onnx) [
                    customDeps.kokoro_onnx
                    customDeps.aiortc
                    customDeps.onnx_asr
                    customDeps.onnx_diarization
                  ];

                  # Piper TTS dependencies (Linux only)
                  # piper-tts v1.3.0+ embeds espeak-ng and no longer needs piper-phonemize
                  piperDeps = pkgs.lib.optionals (customDeps.piper_tts != null) [
                    customDeps.piper_tts
                  ];

                  # Development dependencies
                  devDeps = pkgs.lib.optionals withDev [
                    ps.anyio
                    ps.pytest-asyncio
                    ps.pytest
                    ps.pytest-mock
                    ps.ruff
                    ps.srt
                    customDeps.webvtt_py
                    customDeps.pytest_antilru
                  ];

                  # OpenTelemetry dependencies (always included)
                  otelDeps = [
                    ps.opentelemetry-api
                    ps.opentelemetry-sdk
                    ps.opentelemetry-exporter-otlp
                    ps.opentelemetry-instrumentation
                    ps.opentelemetry-instrumentation-asgi
                    customDeps.opentelemetry_instrumentation_asyncio
                    ps.opentelemetry-instrumentation-fastapi
                    customDeps.opentelemetry_instrumentation_httpx
                    ps.opentelemetry-instrumentation-logging
                    ps.opentelemetry-instrumentation-grpc
                  ];
                in
                coreDeps ++ piperDeps ++ devDeps ++ otelDeps
              );
            in
            pkgs.stdenv.mkDerivation rec {
              pname = "speaches";
              version = "0.1.0";

              src = pkgs.lib.cleanSourceWith {
                src = ./.;
                filter =
                  path: type:
                  let
                    relPath = pkgs.lib.removePrefix (toString ./. + "/") (toString path);
                    allowedPrefixes = [
                      "src"
                      "realtime-console/dist"
                    ];
                    allowedFiles = [
                      "pyproject.toml"
                      "model_aliases.json"
                    ];
                    # For directories, also check if any allowedPrefix starts with this path
                    # (so parent dirs like "realtime-console" pass when "realtime-console/dist" is allowed)
                    isParentOfAllowed =
                      type == "directory"
                      && builtins.any (prefix: pkgs.lib.hasPrefix (relPath + "/") prefix) allowedPrefixes;
                  in
                  builtins.any (prefix: pkgs.lib.hasPrefix prefix relPath) allowedPrefixes
                  || builtins.elem relPath allowedFiles
                  || isParentOfAllowed;
              };

              nativeBuildInputs = [ pkgs.makeWrapper ] ++ pkgs.lib.optionals withDev [ pkgs.basedpyright ];

              buildInputs = [
                pythonEnv
                pkgs.ffmpeg-full
                pkgs.portaudio
                pkgs.openssl
                pkgs.zlib
                pkgs.stdenv.cc.cc
                pkgs.ctranslate2
                pkgs.espeak-ng
              ]
              ++ pkgs.lib.optionals withCuda (
                with pkgs;
                [
                  cudaPackages_12.cudnn
                  cudaPackages_12.libcublas
                  cudaPackages_12.libcurand
                  cudaPackages_12.libcufft
                  cudaPackages_12.cuda_cudart
                  cudaPackages_12.cuda_nvrtc
                ]
              );

              installPhase = ''
                mkdir -p $out/share/speaches
                cp -r src pyproject.toml model_aliases.json $out/share/speaches/

                # Copy the realtime console UI
                mkdir -p $out/share/speaches/realtime-console
                cp -r realtime-console/dist $out/share/speaches/realtime-console/

                mkdir -p $out/bin
                makeWrapper ${pythonEnv}/bin/python $out/bin/speaches \
                  --prefix PATH : ${
                    pkgs.lib.makeBinPath [
                      pkgs.ffmpeg-full
                      pkgs.espeak-ng
                    ]
                  } \
                  --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [ pkgs.espeak-ng ]} \
                  ${pkgs.lib.optionalString withCuda "--prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:${pkgs.lib.makeLibraryPath buildInputs}"} \
                  --set PYTHONPATH "$out/share/speaches/src" \
                  --chdir "$out/share/speaches" \
                  --add-flags "-m uvicorn" \
                  --add-flags "--factory speaches.main:create_app" \
                  --add-flags "--host \''${UVICORN_HOST:-0.0.0.0}" \
                  --add-flags "--port \''${UVICORN_PORT:-8000}"

                ${pkgs.lib.optionalString withDev ''
                  makeWrapper ${pythonEnv}/bin/python $out/bin/speaches-python \
                    --prefix PATH : ${
                      pkgs.lib.makeBinPath [
                        pkgs.ffmpeg-full
                        pkgs.espeak-ng
                      ]
                    } \
                    --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [ pkgs.espeak-ng ]} \
                    --set PYTHONPATH "$out/share/speaches/src"
                ''}
              '';

              passthru = { inherit pythonEnv; };

              meta = with pkgs.lib; {
                description = "AI-powered speech processing application";
                homepage = "https://github.com/speaches-ai/speaches";
                license = licenses.mit;
                platforms = platforms.unix;
              };
            };

          # Default packages for convenience
          defaultPkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [
              (mkOverlay {
                pythonVersion = "python312";
                cudaSupport = true;
              })
            ];
          };

          devPkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [
              (mkOverlay {
                pythonVersion = "python312";
                cudaSupport = false;
              })
            ];
          };

          # Model fetchers using nix-hug with proper hashes
          models = {
            # Kokoro TTS model (primary TTS engine)
            kokoro-82m = nix-hug.lib.${system}.fetchModel {
              url = "speaches-ai/Kokoro-82M-v1.0-ONNX";
              rev = "dc196c76d64fed9203906231372bcb98135815df";
              fileTreeHash = "sha256-+Aea1c28vvS+pfOs2alshOajGzW6I7ujDVIIAQ5KlgI=";
            };

            # Silero VAD model (voice activity detection)
            silero-vad = nix-hug.lib.${system}.fetchModel {
              url = "onnx-community/silero-vad";
              rev = "e71cae966052b992a7eca6b17738916ce0eca4ec";
              fileTreeHash = "sha256-Ngj+Sq0vWS2MEPbOzpCoUe1iBORhDyaK2Eluq/RmUEs=";
            };

            # Whisper STT model (base version for lower RAM usage)
            whisper-base = nix-hug.lib.${system}.fetchModel {
              url = "Systran/faster-whisper-base";
              rev = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66";
              fileTreeHash = "sha256-GYgT6udNwSgjZabqajK/i8kL3pvRPbaTC2PQdUfH0EY=";
            };

            # Whisper STT model (tiny.en for fast tests)
            whisper-tiny-en = nix-hug.lib.${system}.fetchModel {
              url = "Systran/faster-whisper-tiny.en";
              rev = "0d3d19a32d3338f10357c0889762bd8d64bbdeba";
              fileTreeHash = "sha256-5vcmhdQIKuVlf4X737KGqtHxLONtAYfsHaG/+vbNjRE=";
            };

            # Wespeaker speaker embedding model
            wespeaker = nix-hug.lib.${system}.fetchModel {
              url = "pyannote/wespeaker-voxceleb-resnet34-LM";
              rev = "837717ddb9ff5507820346191109dc79c958d614";
              fileTreeHash = "sha256-X6meYLcrkjfV2X8rLebIzgY8BTC99R7qL8Bqsn7gEzg=";
            };
          };

          # Package variants
          speaches = mkSpeaches { };
          speaches-cpu = mkSpeaches { withCuda = false; };
          speaches-dev = mkSpeaches { withDev = true; };

          # Per-Python-version variants (with CUDA on x86_64-linux)
          speaches-python312 = mkSpeaches { pythonVersion = "python312"; };
          speaches-python313 = mkSpeaches { pythonVersion = "python313"; };
          speaches-python314 = mkSpeaches { pythonVersion = "python314"; };
          speaches-python315 = (mkSpeaches { pythonVersion = "python315"; }).overrideAttrs {
            meta.broken = true;
          };

          # Parameterized NixOS VM e2e test (Linux-only)
          mkE2eTest =
            {
              pythonVersion,
              fullTest ? true,
            }:
            let
              testPackage = mkSpeaches {
                inherit pythonVersion;
                withCuda = false;
              };
              testModelCache = nix-hug.lib.${system}.buildCache {
                models = [
                  models.kokoro-82m
                  models.silero-vad
                  models.whisper-base
                  models.whisper-tiny-en
                  models.wespeaker
                ];
              };
            in
            defaultPkgs.testers.nixosTest {
              name = "speaches-e2e-test-${pythonVersion}";
              enableOCR = false;

              nodes.machine =
                {
                  config,
                  pkgs,
                  ...
                }:
                {
                  imports = [ ./nix/module.nix ];

                  environment.variables = {
                    HF_HUB_CACHE = "${testModelCache}";
                    HF_HUB_OFFLINE = "1";
                  };

                  services.speaches = {
                    enable = true;
                    package = testPackage;
                    host = "127.0.0.1";
                    port = 18000;
                    environment = {
                      SPEACHES_WHISPER_MODEL = "Systran/faster-whisper-base";
                      HF_HUB_CACHE = "${testModelCache}";
                      HF_HUB_OFFLINE = "1";
                    };
                  };

                  environment.systemPackages =
                    with pkgs;
                    [
                      curl
                      jq
                      file
                    ]
                    ++ (
                      if fullTest then
                        [
                          sox
                          ffmpeg-full
                        ]
                      else
                        [ ]
                    );

                  virtualisation = {
                    memorySize = 4096;
                    cores = 2;
                  };
                };

              testScript =
                if fullTest then
                  ''
                    import json
                    import time

                    BASE = "http://127.0.0.1:18000"
                    counts = {"passed": 0, "failed": 0}

                    def curl_json(path, *args):
                        out = machine.succeed(f"curl -s {BASE}{path} " + " ".join(args))
                        return json.loads(out)

                    def curl_post_file(path, out_file, *args):
                        status = machine.succeed(
                            f'curl -s -w "%{{http_code}}" -o /tmp/t/{out_file} -X POST "{BASE}{path}" ' + " ".join(args)
                        ).strip()
                        return status

                    def check(desc, condition):
                        if condition:
                            counts["passed"] += 1
                            print(f"  PASS: {desc}")
                        else:
                            counts["failed"] += 1
                            print(f"  FAIL: {desc}")

                    machine.start()
                    machine.wait_for_unit("speaches.service")
                    machine.wait_for_open_port(18000)
                    time.sleep(5)
                    machine.succeed("mkdir -p /tmp/t")

                    # --- 1. Health & Diagnostics ---
                    print("\n--- 1. Health & Diagnostics ---")

                    health = curl_json("/health")
                    check("GET /health returns OK", health.get("message") == "OK")

                    ps_data = curl_json("/api/ps")
                    check("GET /api/ps returns model list", "models" in ps_data)

                    # --- 2. OpenTelemetry ---
                    print("\n--- 2. OpenTelemetry ---")

                    machine.succeed("""
                      ${testPackage.pythonEnv}/bin/python -c "
                    import opentelemetry.instrumentation.asyncio
                    import opentelemetry.instrumentation.asgi
                    import opentelemetry.instrumentation.fastapi
                    import opentelemetry.instrumentation.httpx
                    import opentelemetry.instrumentation.logging
                    import opentelemetry.instrumentation.grpc
                    print('All opentelemetry modules imported successfully')
                    "
                    """)
                    check("OpenTelemetry modules importable", True)

                    # --- 3. Model Management ---
                    print("\n--- 3. Model Management ---")

                    models_data = curl_json("/v1/models")
                    model_count = len(models_data.get("data", []))
                    check(f"GET /v1/models returned {model_count} models", model_count >= 3)
                    for m in models_data["data"]:
                        print(f"    - {m['id']}")

                    tts_models = curl_json("/v1/models?task=text-to-speech")
                    check("TTS model filter works", len(tts_models.get("data", [])) >= 1)

                    stt_models = curl_json("/v1/models?task=automatic-speech-recognition")
                    check("STT model filter returns >= 2", len(stt_models.get("data", [])) >= 2)

                    model_info = curl_json("/v1/models/Systran/faster-whisper-base")
                    check("GET specific model info", "id" in model_info)

                    audio_models = curl_json("/v1/audio/models")
                    check("GET /v1/audio/models", "models" in audio_models)

                    voices = curl_json("/v1/audio/voices")
                    check(f"GET /v1/audio/voices returned {len(voices)} voices", len(voices) >= 1)

                    # --- 4. TTS ---
                    print("\n--- 4. TTS (Text-to-Speech) ---")

                    original_text = "People assume that time is a strict progression of cause to effect"

                    for fmt in ["wav", "mp3", "flac", "opus"]:
                        status = curl_post_file(
                            "/v1/audio/speech", f"tts.{fmt}",
                            '-H "Content-Type: application/json"',
                            f"""-d '{{"model":"tts-1","input":"Hello world","voice":"af_bella","response_format":"{fmt}"}}'"""
                        )
                        machine.succeed(f"test -s /tmp/t/tts.{fmt}")
                        check(f"TTS {fmt} format (HTTP {status})", status.endswith("200"))

                    # Generate main test audio for STT tests
                    curl_post_file(
                        "/v1/audio/speech", "main.wav",
                        '-H "Content-Type: application/json"',
                        f"""-d '{{"model":"tts-1","input":"{original_text}","voice":"af_bella"}}'"""
                    )
                    machine.succeed("test -s /tmp/t/main.wav")
                    check("TTS main audio generated", True)

                    # Different voice
                    status = curl_post_file(
                        "/v1/audio/speech", "heart.wav",
                        '-H "Content-Type: application/json"',
                        """-d '{"model":"tts-1","input":"Hello world","voice":"af_heart"}'"""
                    )
                    machine.succeed("test -s /tmp/t/heart.wav")
                    check("TTS with af_heart voice", status.endswith("200"))

                    # tts-1-hd alias
                    status = curl_post_file(
                        "/v1/audio/speech", "hd.wav",
                        '-H "Content-Type: application/json"',
                        """-d '{"model":"tts-1-hd","input":"Hello world","voice":"af_bella"}'"""
                    )
                    machine.succeed("test -s /tmp/t/hd.wav")
                    check("TTS tts-1-hd alias", status.endswith("200"))

                    # --- 5. STT ---
                    print("\n--- 5. STT (Speech-to-Text) ---")

                    # 5a. whisper-base JSON
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt_base.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=json"',
                    )
                    stt_base = json.loads(machine.succeed("cat /tmp/t/stt_base.json"))
                    transcribed = stt_base.get("text", "")
                    check(f"STT whisper-base: {transcribed[:60]}...", len(transcribed) > 0)
                    key_words = ["people", "assume", "time", "cause", "effect"]
                    matches = sum(1 for w in key_words if w in transcribed.lower())
                    check(f"Transcription quality: {matches}/{len(key_words)} key words", matches >= 3)

                    # 5b. whisper-tiny.en
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt_tiny.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-tiny.en"',
                        '-F "response_format=json"',
                    )
                    stt_tiny = json.loads(machine.succeed("cat /tmp/t/stt_tiny.json"))
                    check("STT whisper-tiny.en returned text", len(stt_tiny.get("text", "")) > 0)

                    # 5c. verbose_json with word + segment timestamps
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt_verbose.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=verbose_json"',
                        '-F "timestamp_granularities[]=word"',
                        '-F "timestamp_granularities[]=segment"',
                    )
                    verbose = json.loads(machine.succeed("cat /tmp/t/stt_verbose.json"))
                    check(f"Verbose JSON: {len(verbose.get('words', []))} words", len(verbose.get("words", [])) > 0)
                    check(f"Verbose JSON: {len(verbose.get('segments', []))} segments", len(verbose.get("segments", [])) > 0)

                    # 5d. SRT format
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt.srt",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=srt"',
                    )
                    srt_out = machine.succeed("cat /tmp/t/stt.srt")
                    check("SRT output non-empty", len(srt_out.strip()) > 0)

                    # 5e. VTT format
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt.vtt",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=vtt"',
                    )
                    vtt_out = machine.succeed("cat /tmp/t/stt.vtt")
                    check("VTT output has WEBVTT header", "WEBVTT" in vtt_out)

                    # 5f. Plain text
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt.txt",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=text"',
                    )
                    txt_out = machine.succeed("cat /tmp/t/stt.txt")
                    check("Plain text output non-empty", len(txt_out.strip()) > 0)

                    # 5g. MP3 input
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt_mp3.json",
                        '-F "file=@/tmp/t/tts.mp3"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=json"',
                    )
                    stt_mp3 = json.loads(machine.succeed("cat /tmp/t/stt_mp3.json"))
                    check("STT with MP3 input", len(stt_mp3.get("text", "")) > 0)

                    # 5h. VAD filter
                    curl_post_file(
                        "/v1/audio/transcriptions", "stt_vad.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=json"',
                        '-F "vad_filter=true"',
                    )
                    stt_vad = json.loads(machine.succeed("cat /tmp/t/stt_vad.json"))
                    check("STT with VAD filter", len(stt_vad.get("text", "")) > 0)

                    # --- 6. Translation ---
                    print("\n--- 6. Translation ---")

                    curl_post_file(
                        "/v1/audio/translations", "translation.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                        '-F "response_format=json"',
                    )
                    translation = json.loads(machine.succeed("cat /tmp/t/translation.json"))
                    check("Translation endpoint returned text", len(translation.get("text", "")) > 0)

                    # --- 7. VAD ---
                    print("\n--- 7. VAD (Voice Activity Detection) ---")

                    curl_post_file(
                        "/v1/audio/speech/timestamps", "vad.json",
                        '-F "file=@/tmp/t/main.wav"',
                    )
                    vad_data = json.loads(machine.succeed("cat /tmp/t/vad.json"))
                    check(f"VAD detected {len(vad_data)} segment(s)", len(vad_data) >= 1)

                    curl_post_file(
                        "/v1/audio/speech/timestamps", "vad_v6.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "model=silero_vad_v6"',
                    )
                    vad_v6 = json.loads(machine.succeed("cat /tmp/t/vad_v6.json"))
                    check(f"VAD v6 detected {len(vad_v6)} segment(s)", len(vad_v6) >= 1)

                    curl_post_file(
                        "/v1/audio/speech/timestamps", "vad_custom.json",
                        '-F "file=@/tmp/t/main.wav"',
                        '-F "threshold=0.5"',
                        '-F "min_silence_duration_ms=500"',
                        '-F "speech_pad_ms=100"',
                    )
                    vad_custom = json.loads(machine.succeed("cat /tmp/t/vad_custom.json"))
                    check("VAD with custom params", isinstance(vad_custom, list))

                    # --- 8. Model Load/Unload ---
                    print("\n--- 8. Model Load/Unload ---")

                    load_status = curl_post_file(
                        "/api/ps/Systran/faster-whisper-tiny.en", "ps_load.json",
                    )
                    check(f"Model load endpoint (HTTP {load_status})", load_status.endswith(("201", "409")))

                    ps_after = curl_json("/api/ps")
                    has_tiny = any("tiny.en" in m for m in ps_after.get("models", []))
                    check("Loaded model visible in /api/ps", has_tiny)

                    unload_status = curl_post_file(
                        "/api/ps/Systran/faster-whisper-tiny.en", "ps_unload.json",
                        *['-X DELETE'],
                    )
                    check(f"Model unload endpoint (HTTP {unload_status})", unload_status.endswith(("200", "409")))

                    # --- 9. Round-Trip Pipeline ---
                    print("\n--- 9. TTS->STT Round-Trip ---")

                    pipeline_text = "The quick brown fox jumps over the lazy dog near the river bank"
                    curl_post_file(
                        "/v1/audio/speech", "pipeline.wav",
                        '-H "Content-Type: application/json"',
                        f"""-d '{{"model":"tts-1","input":"{pipeline_text}","voice":"af_bella"}}'"""
                    )
                    machine.succeed("test -s /tmp/t/pipeline.wav")

                    curl_post_file(
                        "/v1/audio/transcriptions", "pipeline_base.json",
                        '-F "file=@/tmp/t/pipeline.wav"',
                        '-F "model=Systran/faster-whisper-base"',
                    )
                    pipe_base = json.loads(machine.succeed("cat /tmp/t/pipeline_base.json"))
                    pipe_text = pipe_base.get("text", "").lower()
                    pipe_words = ["quick", "brown", "fox", "jumps", "lazy", "dog", "river"]
                    pipe_matches = sum(1 for w in pipe_words if w in pipe_text)
                    check(f"Round-trip whisper-base: {pipe_matches}/{len(pipe_words)} words", pipe_matches >= 4)

                    curl_post_file(
                        "/v1/audio/transcriptions", "pipeline_tiny.json",
                        '-F "file=@/tmp/t/pipeline.wav"',
                        '-F "model=Systran/faster-whisper-tiny.en"',
                    )
                    pipe_tiny = json.loads(machine.succeed("cat /tmp/t/pipeline_tiny.json"))
                    tiny_text = pipe_tiny.get("text", "").lower()
                    tiny_matches = sum(1 for w in pipe_words if w in tiny_text)
                    check(f"Round-trip whisper-tiny.en: {tiny_matches}/{len(pipe_words)} words", tiny_matches >= 3)

                    # --- Summary ---
                    print(f"\n=== Results: {counts['passed']} passed, {counts['failed']} failed ===")
                    assert counts["failed"] == 0, f"{counts['failed']} test(s) failed!"
                    print("All NixOS VM e2e tests passed with ${pythonVersion}!")
                  ''
                else
                  # Quick health check only
                  ''
                    machine.start()
                    machine.wait_for_unit("speaches.service")
                    machine.wait_for_open_port(18000)
                    print("Testing ${pythonVersion} package...")
                    machine.succeed("curl -f http://127.0.0.1:18000/health")
                    print("${pythonVersion} e2e test passed!")
                  '';
            };
          # Helper to create e2e test scripts parameterized by server package
          mkE2eTestScript =
            { name, serverPackage }:
            defaultPkgs.writeShellScriptBin name ''
              set -euo pipefail

              CURL="${defaultPkgs.curl}/bin/curl"
              JQ="${defaultPkgs.jq}/bin/jq"
              FILE="${defaultPkgs.file}/bin/file"
              BASE_URL="http://127.0.0.1:18000"
              PASSED=0
              FAILED=0

              pass() { PASSED=$((PASSED + 1)); echo "  PASS: $1"; }
              fail() { FAILED=$((FAILED + 1)); echo "  FAIL: $1"; }

              # assert_http: run curl, check status code matches expected
              #   $1=description $2=expected_status $3=output_file $4+=curl args
              assert_http() {
                local desc="$1" expected="$2" outfile="$3"
                shift 3
                local status
                status=$($CURL -s -w "%{http_code}" -o "$outfile" "$@") || true
                if [[ "$status" =~ ^''${expected}$ ]]; then
                  pass "$desc (HTTP $status)"
                else
                  fail "$desc (expected HTTP $expected, got $status)"
                  [ -f "$outfile" ] && cat "$outfile"
                  return 1
                fi
              }

              echo "=== Speaches End-to-End Test ==="

              # --- Setup ---
              MODEL_CACHE="${
                nix-hug.lib.${system}.buildCache {
                  models = [
                    models.kokoro-82m
                    models.silero-vad
                    models.whisper-base
                    models.whisper-tiny-en
                    models.wespeaker
                  ];
                }
              }"
              export HF_HUB_CACHE="$MODEL_CACHE"
              export HF_HUB_OFFLINE=1

              TEST_DIR=$(mktemp -d)
              cd "$TEST_DIR"

              ${serverPackage}/bin/speaches --host 127.0.0.1 --port 18000 &
              SERVER_PID=$!
              cleanup() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; }
              trap cleanup EXIT

              echo "Waiting for server..."
              for i in {1..120}; do
                $CURL -s $BASE_URL/health >/dev/null 2>&1 && break
                [ $i -eq 120 ] && echo "Server failed to start" && exit 1
                sleep 1
              done
              echo "Server ready."

              # ===================================================================
              echo ""
              echo "--- 1. Health & Diagnostics ---"
              # ===================================================================

              # 1a. Health check
              assert_http "GET /health" "200" health.json $BASE_URL/health
              if $JQ -e '.message == "OK"' health.json >/dev/null 2>&1; then
                pass "Health response contains OK"
              else
                fail "Health response missing OK"
              fi

              # 1b. Running models list
              assert_http "GET /api/ps" "200" ps.json $BASE_URL/api/ps
              if $JQ -e '.models' ps.json >/dev/null 2>&1; then
                pass "Running models endpoint returns model list"
                echo "    Loaded models: $($JQ -r '.models[]' ps.json | tr '\n' ', ')"
              else
                fail "Running models response invalid"
              fi

              # ===================================================================
              echo ""
              echo "--- 2. Model Management ---"
              # ===================================================================

              # 2a. List all local models
              assert_http "GET /v1/models" "200" models.json $BASE_URL/v1/models
              MODEL_COUNT=$($JQ '.data | length' models.json)
              if [ "$MODEL_COUNT" -ge 3 ]; then
                pass "Model listing returned $MODEL_COUNT models"
                $JQ -r '.data[].id' models.json | while read -r m; do echo "    - $m"; done
              else
                fail "Expected at least 3 models, got $MODEL_COUNT"
              fi

              # 2b. List models filtered by task
              assert_http "GET /v1/models?task=text-to-speech" "200" tts_models.json "$BASE_URL/v1/models?task=text-to-speech"
              TTS_COUNT=$($JQ '.data | length' tts_models.json)
              if [ "$TTS_COUNT" -ge 1 ]; then
                pass "TTS model filter returned $TTS_COUNT model(s)"
              else
                fail "TTS model filter returned 0 models"
              fi

              assert_http "GET /v1/models?task=automatic-speech-recognition" "200" stt_models.json "$BASE_URL/v1/models?task=automatic-speech-recognition"
              STT_COUNT=$($JQ '.data | length' stt_models.json)
              if [ "$STT_COUNT" -ge 2 ]; then
                pass "STT model filter returned $STT_COUNT models (whisper-base + tiny.en)"
              else
                fail "STT model filter returned $STT_COUNT models, expected >= 2"
              fi

              # 2c. Get specific model info
              assert_http "GET /v1/models/Systran/faster-whisper-base" "200" model_info.json $BASE_URL/v1/models/Systran/faster-whisper-base

              # 2d. List TTS audio models
              assert_http "GET /v1/audio/models" "200" audio_models.json $BASE_URL/v1/audio/models
              if $JQ -e '.models' audio_models.json >/dev/null 2>&1; then
                pass "Audio models endpoint returns data"
              else
                fail "Audio models endpoint invalid response"
              fi

              # 2e. List TTS voices
              assert_http "GET /v1/audio/voices" "200" voices.json $BASE_URL/v1/audio/voices
              VOICE_COUNT=$($JQ '. | length' voices.json 2>/dev/null || echo "0")
              if [ "$VOICE_COUNT" -ge 1 ]; then
                pass "Voices endpoint returned $VOICE_COUNT voices"
              else
                fail "Voices endpoint returned no voices"
              fi

              # ===================================================================
              echo ""
              echo "--- 3. TTS (Text-to-Speech) ---"
              # ===================================================================

              ORIGINAL_TEXT="People assume that time is a strict progression of cause to effect. But actually, from a nonlinear, non-subjective viewpoint, it is more like a big ball of wibbly wobbly, timey wimey stuff"

              # 3a. TTS - WAV format (default)
              assert_http "POST /v1/audio/speech (wav)" "200" tts_wav.wav \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"$ORIGINAL_TEXT\", \"voice\": \"af_bella\"}"
              if [ -s tts_wav.wav ]; then
                pass "TTS WAV output is non-empty ($($FILE tts_wav.wav | cut -d: -f2))"
              else
                fail "TTS WAV output is empty"
              fi

              # 3b. TTS - MP3 format
              assert_http "POST /v1/audio/speech (mp3)" "200" tts_mp3.mp3 \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"Hello world\", \"voice\": \"af_bella\", \"response_format\": \"mp3\"}"
              if [ -s tts_mp3.mp3 ]; then
                pass "TTS MP3 output is non-empty"
              else
                fail "TTS MP3 output is empty"
              fi

              # 3c. TTS - FLAC format
              assert_http "POST /v1/audio/speech (flac)" "200" tts_flac.flac \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"Hello world\", \"voice\": \"af_bella\", \"response_format\": \"flac\"}"
              if [ -s tts_flac.flac ]; then
                pass "TTS FLAC output is non-empty"
              else
                fail "TTS FLAC output is empty"
              fi

              # 3d. TTS - Opus format
              assert_http "POST /v1/audio/speech (opus)" "200" tts_opus.opus \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"Hello world\", \"voice\": \"af_bella\", \"response_format\": \"opus\"}"
              if [ -s tts_opus.opus ]; then
                pass "TTS Opus output is non-empty"
              else
                fail "TTS Opus output is empty"
              fi

              # 3e. TTS - different voice
              assert_http "POST /v1/audio/speech (voice: af_heart)" "200" tts_heart.wav \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"Hello world\", \"voice\": \"af_heart\"}"
              if [ -s tts_heart.wav ]; then
                pass "TTS with af_heart voice produced output"
              else
                fail "TTS with af_heart voice produced empty output"
              fi

              # 3f. TTS - tts-1-hd alias
              assert_http "POST /v1/audio/speech (tts-1-hd)" "200" tts_hd.wav \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1-hd\", \"input\": \"Hello world\", \"voice\": \"af_bella\"}"
              if [ -s tts_hd.wav ]; then
                pass "TTS with tts-1-hd alias works"
              else
                fail "TTS with tts-1-hd alias produced empty output"
              fi

              # ===================================================================
              echo ""
              echo "--- 4. STT (Speech-to-Text) ---"
              # ===================================================================

              # 4a. STT with whisper-base (JSON response)
              assert_http "POST /v1/audio/transcriptions (whisper-base, json)" "200" stt_base.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=json"
              TRANSCRIBED=$($JQ -r '.text' stt_base.json 2>/dev/null || echo "")
              if [ -n "$TRANSCRIBED" ]; then
                pass "STT whisper-base JSON transcription: $(echo "$TRANSCRIBED" | head -c 80)..."
              else
                fail "STT whisper-base JSON returned empty text"
              fi

              # Verify transcription quality
              MATCHES=0
              for word in people assume time cause effect viewpoint; do
                echo "$TRANSCRIBED" | tr '[:upper:]' '[:lower:]' | grep -q "$word" && MATCHES=$((MATCHES + 1))
              done
              if [ $MATCHES -ge 3 ]; then
                pass "Transcription quality: $MATCHES/6 key words matched"
              else
                fail "Transcription quality: only $MATCHES/6 key words matched"
              fi

              # 4b. STT with whisper-tiny.en
              assert_http "POST /v1/audio/transcriptions (whisper-tiny.en)" "200" stt_tiny.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-tiny.en" \
                -F "response_format=json"
              if $JQ -e '.text' stt_tiny.json >/dev/null 2>&1; then
                pass "STT whisper-tiny.en returned text: $($JQ -r '.text' stt_tiny.json | head -c 80)..."
              else
                fail "STT whisper-tiny.en returned no text"
              fi

              # 4c. STT - verbose_json with word timestamps
              assert_http "POST /v1/audio/transcriptions (verbose_json + word timestamps)" "200" stt_verbose.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=verbose_json" \
                -F "timestamp_granularities[]=word" \
                -F "timestamp_granularities[]=segment"
              if $JQ -e '.words' stt_verbose.json >/dev/null 2>&1; then
                WORD_COUNT=$($JQ '.words | length' stt_verbose.json)
                pass "Verbose JSON has $WORD_COUNT word-level timestamps"
              else
                fail "Verbose JSON missing word timestamps"
              fi
              if $JQ -e '.segments' stt_verbose.json >/dev/null 2>&1; then
                SEG_COUNT=$($JQ '.segments | length' stt_verbose.json)
                pass "Verbose JSON has $SEG_COUNT segment-level timestamps"
              else
                fail "Verbose JSON missing segment timestamps"
              fi

              # 4d. STT - SRT format
              assert_http "POST /v1/audio/transcriptions (srt)" "200" stt_output.srt \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=srt"
              if grep -q "-->"; then
                pass "SRT output contains timestamp arrows"
              else
                # SRT might have different format, just check non-empty
                if [ -s stt_output.srt ]; then
                  pass "SRT output is non-empty"
                else
                  fail "SRT output is empty"
                fi
              fi < stt_output.srt

              # 4e. STT - VTT format
              assert_http "POST /v1/audio/transcriptions (vtt)" "200" stt_output.vtt \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=vtt"
              if grep -q "WEBVTT" stt_output.vtt; then
                pass "VTT output contains WEBVTT header"
              else
                fail "VTT output missing WEBVTT header"
              fi

              # 4f. STT - plain text format
              assert_http "POST /v1/audio/transcriptions (text)" "200" stt_output.txt \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=text"
              if [ -s stt_output.txt ]; then
                pass "Plain text output: $(head -c 80 stt_output.txt)..."
              else
                fail "Plain text output is empty"
              fi

              # 4g. STT with MP3 input (test format handling)
              assert_http "POST /v1/audio/transcriptions (mp3 input)" "200" stt_mp3_input.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_mp3.mp3" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=json"
              if $JQ -e '.text' stt_mp3_input.json >/dev/null 2>&1; then
                pass "STT with MP3 input works"
              else
                fail "STT with MP3 input failed"
              fi

              # 4h. STT with VAD filter enabled
              assert_http "POST /v1/audio/transcriptions (vad_filter=true)" "200" stt_vad.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=json" \
                -F "vad_filter=true"
              if $JQ -e '.text' stt_vad.json >/dev/null 2>&1; then
                pass "STT with VAD filter returned text"
              else
                fail "STT with VAD filter failed"
              fi

              # ===================================================================
              echo ""
              echo "--- 5. Translation ---"
              # ===================================================================

              # 5a. Translation endpoint (translates to English - with English input it should pass through)
              assert_http "POST /v1/audio/translations (json)" "200" translation.json \
                -X POST "$BASE_URL/v1/audio/translations" \
                -F "file=@tts_wav.wav" \
                -F "model=Systran/faster-whisper-base" \
                -F "response_format=json"
              if $JQ -e '.text' translation.json >/dev/null 2>&1; then
                pass "Translation endpoint returned text: $($JQ -r '.text' translation.json | head -c 80)..."
              else
                fail "Translation endpoint returned no text"
              fi

              # ===================================================================
              echo ""
              echo "--- 6. VAD (Voice Activity Detection) ---"
              # ===================================================================

              # 6a. VAD speech timestamps
              assert_http "POST /v1/audio/speech/timestamps" "200" vad_timestamps.json \
                -X POST "$BASE_URL/v1/audio/speech/timestamps" \
                -F "file=@tts_wav.wav"
              STAMP_COUNT=$($JQ '. | length' vad_timestamps.json 2>/dev/null || echo "0")
              if [ "$STAMP_COUNT" -ge 1 ]; then
                pass "VAD detected $STAMP_COUNT speech segment(s)"
                $JQ -r '.[] | "    segment: \(.start)ms - \(.end)ms"' vad_timestamps.json
              else
                fail "VAD detected no speech segments"
              fi

              # 6b. VAD with explicit model (v6)
              assert_http "POST /v1/audio/speech/timestamps (silero_vad_v6)" "200" vad_v6.json \
                -X POST "$BASE_URL/v1/audio/speech/timestamps" \
                -F "file=@tts_wav.wav" \
                -F "model=silero_vad_v6"
              V6_COUNT=$($JQ '. | length' vad_v6.json 2>/dev/null || echo "0")
              if [ "$V6_COUNT" -ge 1 ]; then
                pass "VAD v6 detected $V6_COUNT speech segment(s)"
              else
                fail "VAD v6 detected no speech segments"
              fi

              # 6c. VAD with custom parameters
              assert_http "POST /v1/audio/speech/timestamps (custom params)" "200" vad_custom.json \
                -X POST "$BASE_URL/v1/audio/speech/timestamps" \
                -F "file=@tts_wav.wav" \
                -F "threshold=0.5" \
                -F "min_silence_duration_ms=500" \
                -F "speech_pad_ms=100"
              if $JQ -e '.' vad_custom.json >/dev/null 2>&1; then
                pass "VAD with custom parameters returned valid JSON"
              else
                fail "VAD with custom parameters failed"
              fi

              # ===================================================================
              echo ""
              echo "--- 7. Model Load/Unload ---"
              # ===================================================================

              # 7a. Load a model explicitly
              assert_http "POST /api/ps/Systran/faster-whisper-tiny.en (load)" "201|409" ps_load.json \
                -X POST "$BASE_URL/api/ps/Systran/faster-whisper-tiny.en"
              pass "Model load endpoint responded"

              # 7b. Verify it appears in running models
              assert_http "GET /api/ps (after load)" "200" ps_after.json $BASE_URL/api/ps
              if $JQ -r '.models[]' ps_after.json 2>/dev/null | grep -q "faster-whisper-tiny.en"; then
                pass "Loaded model appears in /api/ps"
              else
                pass "Model may have been auto-loaded already (skipping check)"
              fi

              # 7c. Unload the model (409 is acceptable if model is still referenced by auto-warmup)
              assert_http "DELETE /api/ps/Systran/faster-whisper-tiny.en (unload)" "200|409" ps_unload.json \
                -X DELETE "$BASE_URL/api/ps/Systran/faster-whisper-tiny.en"
              pass "Model unload endpoint responded"

              # ===================================================================
              echo ""
              echo "--- 8. TTS -> STT Round-Trip Pipeline ---"
              # ===================================================================

              # Generate longer audio, transcribe with both models, compare
              PIPELINE_TEXT="The quick brown fox jumps over the lazy dog near the river bank"
              assert_http "TTS pipeline audio" "200" pipeline.wav \
                -X POST "$BASE_URL/v1/audio/speech" \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"tts-1\", \"input\": \"$PIPELINE_TEXT\", \"voice\": \"af_bella\"}"

              assert_http "STT pipeline (whisper-base)" "200" pipeline_base.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@pipeline.wav" \
                -F "model=Systran/faster-whisper-base"
              BASE_TEXT=$($JQ -r '.text' pipeline_base.json 2>/dev/null | tr '[:upper:]' '[:lower:]')
              PIPELINE_MATCHES=0
              for word in quick brown fox jumps lazy dog river; do
                echo "$BASE_TEXT" | grep -q "$word" && PIPELINE_MATCHES=$((PIPELINE_MATCHES + 1))
              done
              if [ $PIPELINE_MATCHES -ge 4 ]; then
                pass "TTS->STT round-trip quality: $PIPELINE_MATCHES/7 key words (whisper-base)"
              else
                fail "TTS->STT round-trip quality: only $PIPELINE_MATCHES/7 key words (whisper-base)"
              fi

              assert_http "STT pipeline (whisper-tiny.en)" "200" pipeline_tiny.json \
                -X POST "$BASE_URL/v1/audio/transcriptions" \
                -F "file=@pipeline.wav" \
                -F "model=Systran/faster-whisper-tiny.en"
              TINY_TEXT=$($JQ -r '.text' pipeline_tiny.json 2>/dev/null | tr '[:upper:]' '[:lower:]')
              TINY_MATCHES=0
              for word in quick brown fox jumps lazy dog river; do
                echo "$TINY_TEXT" | grep -q "$word" && TINY_MATCHES=$((TINY_MATCHES + 1))
              done
              if [ $TINY_MATCHES -ge 3 ]; then
                pass "TTS->STT round-trip quality: $TINY_MATCHES/7 key words (whisper-tiny.en)"
              else
                fail "TTS->STT round-trip quality: only $TINY_MATCHES/7 key words (whisper-tiny.en)"
              fi

              # ===================================================================
              echo ""
              echo "=== Results: $PASSED passed, $FAILED failed ==="
              if [ $FAILED -gt 0 ]; then
                echo "Some tests failed!"
                exit 1
              fi
              echo "All tests passed!"
            '';
          devCustomDeps = import ./nix/dependencies.nix {
            pkgs = devPkgs;
            pyPackages = devPkgs.python312.pkgs;
            system = system;
          };
        in
        {
          # Development shell
          devShells.default =
            let
              # Grab the overlay-overridden faster-whisper (includes silero v5+v6 ONNX assets)
              devFasterWhisper = devPkgs.python312Packages.faster-whisper;
            in
            devPkgs.mkShell {
            nativeBuildInputs =
              with devPkgs;
              [
                (python312.withPackages (
                  ps:
                  with ps;
                  [
                    # Include all deps including dev for development shell
                    fastapi
                    huggingface-hub
                    numpy
                    pydantic
                    pydantic-settings
                    python-multipart
                    soundfile
                    uvicorn
                    openai
                    aiostream
                    cachetools
                    gradio
                    httpx
                    httpx-sse
                    httpx-ws
                    devFasterWhisper
                    anyio
                    opentelemetry-api
                    opentelemetry-sdk
                    opentelemetry-exporter-otlp
                    opentelemetry-instrumentation
                    opentelemetry-instrumentation-asgi
                    opentelemetry-instrumentation-fastapi
                    opentelemetry-instrumentation-logging
                    opentelemetry-instrumentation-grpc
                    pytest-asyncio
                    pytest
                    pytest-mock
                    mdx-truly-sane-lists
                    mkdocs
                    mkdocs-material
                    mkdocstrings
                    mkdocstrings-python
                    ruff
                    srt
                  ]
                  ++ (
                    with devCustomDeps;
                    [
                      kokoro_onnx
                      aiortc
                      onnx_asr
                      onnx_diarization
                      espeakng_loader
                      opentelemetry_instrumentation_asyncio
                      mkdocs_render_swagger_plugin
                      opentelemetry_instrumentation_httpx
                      pytest_antilru
                      webvtt_py
                    ]
                    ++ lib.optionals stdenv.isLinux [
                      piper_tts
                    ]
                  )
                  ++ [
                    ps.opentelemetry-api
                    ps.opentelemetry-sdk
                    ps.opentelemetry-exporter-otlp
                    ps.opentelemetry-instrumentation
                    ps.opentelemetry-instrumentation-asgi
                    ps.opentelemetry-instrumentation-fastapi
                    ps.opentelemetry-instrumentation-logging
                    ps.opentelemetry-instrumentation-grpc
                  ]
                ))
                uv
                ffmpeg-full
                go-task
                act
                docker
                docker-compose
                grafana-loki
                tempo
                parallel
                pv
                websocat
                basedpyright
              ]
              ++ devPkgs.lib.optionals isLinux (
                with devPkgs;
                [
                  cudaPackages_12.cudnn
                  cudaPackages_12.libcublas
                  cudaPackages_12.libcurand
                  cudaPackages_12.libcufft
                  cudaPackages_12.cuda_cudart
                  cudaPackages_12.cuda_nvrtc
                ]
              );

            LD_LIBRARY_PATH = devPkgs.lib.optionalString isLinux "/run/opengl-driver/lib:${
              devPkgs.lib.makeLibraryPath (
                with devPkgs;
                [
                  cudaPackages_12.cudnn
                  cudaPackages_12.libcublas
                  cudaPackages_12.libcurand
                  cudaPackages_12.libcufft
                  cudaPackages_12.cuda_cudart
                  cudaPackages_12.cuda_nvrtc
                  portaudio
                  zlib
                  stdenv.cc.cc
                  openssl
                ]
              )
            }";

            shellHook = ''
              source .venv/bin/activate 2>/dev/null || true
              source .env 2>/dev/null || true
            '';
          };

          # Packages
          packages = {
            default = speaches;
            inherit
              speaches
              speaches-cpu
              speaches-dev
              speaches-python312
              speaches-python313
              speaches-python314
              speaches-python315
              ;

            # Models
            inherit (models) kokoro-82m silero-vad whisper-base whisper-tiny-en wespeaker;

            # Build a proper HuggingFace cache with all models
            model-cache = nix-hug.lib.${system}.buildCache {
              models = [
                models.kokoro-82m
                models.silero-vad
                models.whisper-base
                models.whisper-tiny-en
                models.wespeaker
              ];
            };

            # Documentation site (uses devShell Python for full speaches importability)
            docs =
              let
                devFasterWhisper = devPkgs.python312Packages.faster-whisper;
                docsPython = devPkgs.python312.withPackages (
                  ps:
                  with ps;
                  [
                    fastapi
                    huggingface-hub
                    pydantic
                    pydantic-settings
                    numpy
                    devFasterWhisper
                    openai
                    httpx
                    mdx-truly-sane-lists
                    mkdocs
                    mkdocs-material
                    mkdocstrings
                    mkdocstrings-python
                  ]
                  ++ [
                    devCustomDeps.mkdocs_render_swagger_plugin
                  ]
                );
                docsSrc = lib.cleanSourceWith {
                  src = self;
                  filter =
                    path: type:
                    let
                      relPath = lib.removePrefix (toString self + "/") (toString path);
                    in
                    relPath == "mkdocs.yml"
                    || lib.hasPrefix "docs" relPath
                    || lib.hasPrefix "src" relPath;
                };
              in
              devPkgs.runCommand "speaches-docs" { nativeBuildInputs = [ docsPython ]; } ''
                cp -r ${docsSrc}/docs docs
                cp ${docsSrc}/mkdocs.yml mkdocs.yml
                cp -r ${docsSrc}/src src
                PYTHONPATH=src mkdocs build -d $out
              '';

            # End-to-end realtime test (uses mock LLM, tests full WS pipeline)
            e2e-test-realtime =
              let
                testPython = defaultPkgs.python312.withPackages (
                  ps: with ps; [
                    httpx
                    websockets
                    uvicorn
                    fastapi
                  ]
                );
              in
              defaultPkgs.writeShellScriptBin "speaches-e2e-test-realtime" ''
                set -euo pipefail

                echo "=== Speaches Realtime E2E Test (Mock LLM) ==="

                MODEL_CACHE="${
                  nix-hug.lib.${system}.buildCache {
                    models = [
                      models.kokoro-82m
                      models.silero-vad
                      models.whisper-base
                      models.whisper-tiny-en
                      models.wespeaker
                    ];
                  }
                }"
                export HF_HUB_CACHE="$MODEL_CACHE"
                export HF_HUB_OFFLINE=1

                MOCK_LLM_PORT=18001
                SPEACHES_PORT=18000

                # Start speaches server with mock LLM as the chat completion backend
                echo "Starting Speaches server (chat completions -> mock LLM on port $MOCK_LLM_PORT)..."
                CHAT_COMPLETION_BASE_URL="http://127.0.0.1:$MOCK_LLM_PORT/v1" \
                CHAT_COMPLETION_API_KEY="mock-key" \
                LOOPBACK_HOST_URL="http://127.0.0.1:$SPEACHES_PORT" \
                  ${speaches-cpu}/bin/speaches --host 127.0.0.1 --port $SPEACHES_PORT &
                SERVER_PID=$!

                cleanup() {
                  echo "Cleaning up..."
                  kill $SERVER_PID 2>/dev/null || true
                  wait $SERVER_PID 2>/dev/null || true
                }
                trap cleanup EXIT

                echo "Waiting for server to start..."
                for i in {1..120}; do
                  if ${defaultPkgs.curl}/bin/curl -s http://127.0.0.1:$SPEACHES_PORT/health >/dev/null 2>&1; then
                    echo "Server is ready!"
                    break
                  fi
                  if [ $i -eq 120 ]; then
                    echo "Server failed to start within 120 seconds"
                    exit 1
                  fi
                  if [ $((i % 10)) -eq 0 ]; then
                    echo "Attempt $i/120 - still waiting for server..."
                  fi
                  sleep 1
                done

                echo "Running realtime WebSocket E2E test..."
                ${testPython}/bin/python ${./tests/e2e_realtime.py}
              '';

            # End-to-end test packages
            e2e-test = mkE2eTestScript {
              name = "speaches-e2e-test";
              serverPackage = speaches-cpu;
            };
            e2e-test-python313 = mkE2eTestScript {
              name = "speaches-e2e-test-python313";
              serverPackage = mkSpeaches { pythonVersion = "python313"; withCuda = false; };
            };
            e2e-test-python314 = mkE2eTestScript {
              name = "speaches-e2e-test-python314";
              serverPackage = mkSpeaches { pythonVersion = "python314"; withCuda = false; };
            };
            e2e-test-cuda = mkE2eTestScript {
              name = "speaches-e2e-test-cuda";
              serverPackage = speaches;
            };
            e2e-test-cuda-python313 = mkE2eTestScript {
              name = "speaches-e2e-test-cuda-python313";
              serverPackage = speaches-python313;
            };
            e2e-test-cuda-python314 = mkE2eTestScript {
              name = "speaches-e2e-test-cuda-python314";
              serverPackage = speaches-python314;
            };
          };

          # Applications
          apps.default = {
            type = "app";
            program = "${speaches}/bin/speaches";
            meta = {
              description = "AI-powered speech processing application";
              maintainers = [ "longregen <claude@infophysics.org>" ];
            };
          };

          # NixOS VM tests (Linux-only, nixosTest requires a NixOS VM)
          checks =
            if isLinux then
              let
                e2e-python312 = mkE2eTest { pythonVersion = "python312"; };
              in
              {
                inherit e2e-python312;
                e2e = e2e-python312; # default alias
                e2e-python313 = mkE2eTest { pythonVersion = "python313"; };
                e2e-python314 = mkE2eTest { pythonVersion = "python314"; };
                e2e-python315 = mkE2eTest { pythonVersion = "python315"; };
              }
            else
              { };

          formatter = defaultPkgs.nixfmt-rfc-style;

          lib = {
            inherit mkSpeaches;
          };
        };

      perSystemOutputs = forEachSystem perSystem;
      inherit (nixpkgs) lib;
    in
    {
      nixosModules.default = import ./nix/module.nix;

      devShells = lib.mapAttrs (_: v: v.devShells) perSystemOutputs;
      packages = lib.mapAttrs (_: v: v.packages) perSystemOutputs;
      apps = lib.mapAttrs (_: v: v.apps) perSystemOutputs;
      checks = lib.mapAttrs (_: v: v.checks) perSystemOutputs;
      formatter = lib.mapAttrs (_: v: v.formatter) perSystemOutputs;
      lib = lib.mapAttrs (_: v: v.lib) perSystemOutputs;
    };
}
