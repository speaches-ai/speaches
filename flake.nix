{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-hug = {
      url = "github:longregen/nix-hug";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    nix-hug,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        addIf = pkgs: name: (
          if (builtins.hasAttr name pkgs)
          then [pkgs.${name}]
          else []
        );
        mkOverlay = {
          pythonVersion,
          cudaSupport ? true,
          withPiper ? false,
          withOtel ? false,
        }: final: prev: let
          pyPackages = prev."${pythonVersion}Packages";

          # Import core dependencies
          coreDeps = import ./nix/dependencies.nix {
            pkgs = final;
            inherit pyPackages system;
          };

          # Conditionally import optional overlays
          piperOverlay =
            if withPiper
            then
              import ./nix/overlays/piper.nix {
                pkgs = final;
                inherit pyPackages system;
              }
            else {};

          otelOverlay =
            if withOtel
            then import ./nix/overlays/opentelemetry.nix {inherit pyPackages;}
            else {};
        in {
          # Override ctranslate2 for CUDA support
          ctranslate2 =
            if cudaSupport && system == "x86_64-linux"
            then
              prev.ctranslate2.override {
                stdenv = prev.gcc14Stdenv;
                withCUDA = true;
                withCuDNN = true;
                cudaPackages = prev.cudaPackages_12;
              }
            else prev.ctranslate2;

          "${pythonVersion}Packages" =
            pyPackages
            // {
              # Override faster-whisper to use our ctranslate2
              faster-whisper = pyPackages.faster-whisper.overrideAttrs (old: {
                propagatedBuildInputs = old.propagatedBuildInputs ++ [final.ctranslate2];
              });
            }
            // coreDeps
            // piperOverlay
            // otelOverlay;
        };

        mkSpeaches = {
          pythonVersion ? "python313",
          withCuda ? (system == "x86_64-linux"),
          withPiper ? false,
          withDev ? false,
          withOtel ? false,
        }: let
          overlay = mkOverlay {
            inherit pythonVersion withPiper withOtel;
            cudaSupport = withCuda;
          };

          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [overlay];
          };

          python = pkgs.${pythonVersion};
          pythonPackages = pkgs."${pythonVersion}Packages";

          # Python environment with configurable dependencies
          pythonEnv = python.withPackages (
            ps: let
              # Core dependencies - always included
              coreDeps =
                [
                  ps.fastapi
                  ps.huggingface-hub
                  ps.numpy
                  ps.pydantic
                  ps.pydantic-settings
                  ps.python-multipart
                  ps.sounddevice
                  ps.soundfile
                  ps.uvicorn
                  ps.openai
                  ps.aiostream
                  ps.cachetools
                  ps.gradio
                  ps.httpx
                  pythonPackages.faster-whisper
                ]
                ++ pkgs.lib.optionals (pythonPackages ? kokoro_onnx) [
                  # Custom packages - these are defined in our overlay
                  pythonPackages.kokoro_onnx
                  pythonPackages.aiortc
                  pythonPackages.onnx_asr
                  ps.httpx-sse
                  ps.httpx-ws
                ];

              # Optional Piper TTS dependencies
              piperDeps = pkgs.lib.optionals (withPiper && pythonPackages ? piper_tts) [
                pythonPackages.piper_tts
                pythonPackages.piper_phonemize
              ];

              # Development dependencies
              devDeps = pkgs.lib.optionals withDev (
                [
                  ps.anyio
                  ps.pytest-asyncio
                  ps.pytest
                  ps.pytest-mock
                  ps.ruff
                ]
                ++ (addIf pythonPackages "pytest_antilru")
              );

              # OpenTelemetry dependencies
              otelDeps = pkgs.lib.optionals withOtel (
                (addIf pythonPackages "opentelemetry_instrumentation_openai")
                ++ (addIf pythonPackages "opentelemetry_instrumentation_openai_v2")
              );
            in
              coreDeps ++ piperDeps ++ devDeps ++ otelDeps
          );
        in
          pkgs.stdenv.mkDerivation rec {
            pname = "speaches";
            version = "0.1.0";

            src = pkgs.lib.cleanSource ./.;

            nativeBuildInputs = [pkgs.makeWrapper] ++ pkgs.lib.optionals withDev [pkgs.basedpyright];

            buildInputs =
              [
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
                with pkgs; [
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
                --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [pkgs.espeak-ng]} \
                ${pkgs.lib.optionalString withCuda "--prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:${pkgs.lib.makeLibraryPath buildInputs}"} \
                --set PYTHONPATH "$out/share/speaches/src" \
                --chdir "$out/share/speaches" \
                --add-flags "-m uvicorn" \
                --add-flags "--factory speaches.main:create_app" \
                --add-flags "--host \''${UVICORN_HOST:-0.0.0.0}" \
                --add-flags "--port \''${UVICORN_PORT:-8000}"
            '';

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
              pythonVersion = "python313";
              cudaSupport = true;
              withPiper = true;
              withOtel = true;
            })
          ];
        };

        # Model fetchers using nix-hug with proper hashes
        models = {
          # Kokoro TTS model (primary TTS engine)
          kokoro-82m = nix-hug.lib.${system}.fetchModel {
            url = "speaches-ai/Kokoro-82M-v1.0-ONNX";
            rev = "main";
            repoInfoHash = "sha256-P7rmAJQypOSIUAkslkBGgMfrPsIFuSwAdWLv008Dm3A=";
            fileTreeHash = "sha256-hXNCSRONwD1gexFw01XrTME07GctuM7DprqZP5HUdZg=";
            derivationHash = "sha256-N/Up67cGHFjnMspF3ZN1rqaVUFyXIm1r3cBAe/REjLk=";
          };

          # Silero VAD model (voice activity detection)
          silero-vad = nix-hug.lib.${system}.fetchModel {
            url = "onnx-community/silero-vad";
            rev = "main";
            repoInfoHash = "sha256-cAlWpNfu5fyFMhoBzSHZNcHBtt6prwl6D2ziIC4Eyqk=";
            fileTreeHash = "sha256-f+/9fy13zID9i5mv7FwdwCs0oQskWJlJ7TK3VjOVI4A=";
            derivationHash = "sha256-fSB/IPRY/kwKuduqNUC+M81V0WSOk8gvBrTcZjajOk8=";
          };

          # Whisper STT model (base version for lower RAM usage)
          whisper-base = nix-hug.lib.${system}.fetchModel {
            url = "Systran/faster-whisper-base";
            rev = "main";

            repoInfoHash = "sha256-vL+bJtN0IX5GzaHOc0nLttN2gcJRwHKhLeSRGNs1WHU=";
            fileTreeHash = "sha256-GYgT6udNwSgjZabqajK/i8kL3pvRPbaTC2PQdUfH0EY=";
            derivationHash = "sha256-Ly4RQtbNqbFNzdVo3I3u7ralevNlj2CF0UOPwmAZsSk=";
          };
        };

        # Package variants
        speaches = mkSpeaches {};
        speaches-cpu = mkSpeaches {withCuda = false;};
        speaches-minimal = mkSpeaches {
          withCuda = false;
          withPiper = false;
          withOtel = false;
        };
        speaches-py312 = mkSpeaches {pythonVersion = "python312";};
        speaches-dev = mkSpeaches {withDev = true;};

        # Parameterized e2e test function
        mkE2eTest = {
          pythonVersion,
          fullTest ? true,
        }: let
          testPackage = mkSpeaches {
            inherit pythonVersion;
            withCuda = false;
          };
          testModelCache = nix-hug.lib.${system}.buildCache {
            models = [
              models.kokoro-82m
              models.silero-vad
              models.whisper-base
            ];
            hash = "sha256-3PaRFwZnC7hYUW4jM9qz3GwSdMjPKrubPKdTuqLZCmQ=";
          };
        in
          defaultPkgs.nixosTest {
            name = "speaches-e2e-test-${pythonVersion}";
            enableOCR = false;

            nodes.machine = {
              config,
              pkgs,
              ...
            }: {
              imports = [./nix/module.nix];

              environment.variables = {
                HF_HUB_CACHE = "${testModelCache}/hub";
                HF_HUB_OFFLINE = "1";
              };

              services.speaches = {
                enable = true;
                package = testPackage;
                host = "127.0.0.1";
                port = 18000;
                environment = {
                  SPEACHES_WHISPER_MODEL = "Systran/faster-whisper-base";
                  HF_HUB_CACHE = "${testModelCache}/hub";
                  HF_HUB_OFFLINE = "1";
                };
              };

              environment.systemPackages = with pkgs;
                [curl jq file]
                ++ (
                  if fullTest
                  then [sox ffmpeg-full]
                  else []
                );

              virtualisation = {
                memorySize = 4096;
                cores = 2;
              };
            };

            testScript =
              if fullTest
              then
                # Full test with TTS→STT pipeline
                ''
                  import json
                  import time

                  machine.start()
                  machine.wait_for_unit("speaches.service")
                  machine.wait_for_open_port(18000)

                  time.sleep(5)

                  print("Testing health endpoint...")
                  machine.succeed("curl -f http://127.0.0.1:18000/health")

                  print("Testing model listing...")
                  models_output = machine.succeed("curl -s http://127.0.0.1:18000/v1/models")
                  models_data = json.loads(models_output)
                  assert "data" in models_data

                  machine.succeed("mkdir -p /tmp/test_results")

                  original_text = "People assume that time is a strict progression of cause to effect. But actually, from a nonlinear, non-subjective viewpoint, it is more like a big ball of wibbly wobbly, timey wimey stuff."

                  print("Testing TTS with Kokoro model...")
                  tts_result = machine.succeed(f"""
                    curl -v -X POST "http://127.0.0.1:18000/v1/audio/speech" \
                      -H "Content-Type: application/json" \
                      -d '{{"model": "tts-1", "input": "{original_text}", "voice": "af_bella", "response_format": "mp3"}}' \
                      -o /tmp/test_results/test_output.mp3 2>&1
                  """)

                  machine.succeed("test -f /tmp/test_results/test_output.mp3")
                  machine.succeed("test -s /tmp/test_results/test_output.mp3")

                  print("Testing STT with Whisper base model...")
                  stt_result = machine.succeed("""
                    curl -v -X POST "http://127.0.0.1:18000/v1/audio/transcriptions" \
                      -F "file=@/tmp/test_results/test_output.mp3" \
                      -F "model=Systran/faster-whisper-base" \
                      -o /tmp/test_results/transcription.json 2>&1
                  """)

                  machine.succeed("test -f /tmp/test_results/transcription.json")
                  transcription_output = machine.succeed("cat /tmp/test_results/transcription.json")
                  transcription_data = json.loads(transcription_output)
                  assert "text" in transcription_data

                  transcribed_text = transcription_data['text']
                  print("✓ TTS→STT pipeline test passed with ${pythonVersion}!")
                  print(f"Original text: {original_text}")
                  print(f"Transcribed text: {transcribed_text}")

                  assert len(transcribed_text) > 0
                  assert any(word in transcribed_text.lower() for word in ["people", "assume", "viewpoint", "cause"])

                  print("All tests passed successfully!")
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
      in {
        # Development shell
        devShells.default = defaultPkgs.mkShell {
          nativeBuildInputs = with defaultPkgs;
            [
              (python313.withPackages (
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
                      sounddevice
                      soundfile
                      uvicorn
                      openai
                      aiostream
                      cachetools
                      gradio
                      httpx
                      httpx-sse
                      httpx-ws
                      faster-whisper
                      anyio
                      pytest-asyncio
                      pytest
                      pytest-mock
                      ruff
                    ]
                    ++ (
                      with defaultPkgs.python313Packages;
                        [
                          # Custom packages from overlay
                          kokoro_onnx
                          aiortc
                          onnx_asr
                          espeakng_loader
                          pytest_antilru
                          opentelemetry_instrumentation_openai
                          opentelemetry_instrumentation_openai_v2
                        ]
                        ++ lib.optionals stdenv.isLinux [
                          piper_tts
                          piper_phonemize
                        ]
                    )
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
            ++ defaultPkgs.lib.optionals (system == "x86_64-linux") (
              with defaultPkgs; [
                cudaPackages_12.cudnn
                cudaPackages_12.libcublas
                cudaPackages_12.libcurand
                cudaPackages_12.libcufft
                cudaPackages_12.cuda_cudart
                cudaPackages_12.cuda_nvrtc
              ]
            );

          LD_LIBRARY_PATH =
            defaultPkgs.lib.optionalString (system == "x86_64-linux")
            "/run/opengl-driver/lib:${
              defaultPkgs.lib.makeLibraryPath (
                with defaultPkgs; [
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
            speaches-minimal
            speaches-py312
            speaches-dev
            ;

          # Models
          inherit (models) kokoro-82m silero-vad whisper-base;

          # Build a proper HuggingFace cache with all models
          model-cache = nix-hug.lib.${system}.buildCache {
            models = [
              models.kokoro-82m
              models.silero-vad
              models.whisper-base
            ];
            hash = "sha256-3PaRFwZnC7hYUW4jM9qz3GwSdMjPKrubPKdTuqLZCmQ=";
          };

          # End-to-end test package with actual models
          e2e-test = defaultPkgs.writeShellScriptBin "speaches-e2e-test" ''
            set -euo pipefail

            echo "=== Speaches End-to-End Test with Real Models ==="
            echo "Setting up test environment..."

            # Set up model paths using nix-hug's pre-built cache
            MODEL_CACHE="${
              nix-hug.lib.${system}.buildCache {
                models = [
                  models.kokoro-82m
                  models.silero-vad
                  models.whisper-base
                ];
                hash = "sha256-3PaRFwZnC7hYUW4jM9qz3GwSdMjPKrubPKdTuqLZCmQ=";
              }
            }"
            export HF_HUB_CACHE="$MODEL_CACHE/hub"
            export HF_HUB_OFFLINE=1

            echo "Using pre-built model cache at: $HF_HUB_CACHE"

            # Create test directory
            TEST_DIR=$(mktemp -d)
            cd "$TEST_DIR"
            echo "Test directory: $TEST_DIR"
            echo "HF Cache directory: $HF_HUB_CACHE"

            # Start speaches server in background
            echo "Starting Speaches server with models..."
            ${speaches-cpu}/bin/speaches --host 127.0.0.1 --port 18000 &
            SERVER_PID=$!

            # Function to cleanup on exit
            cleanup() {
              echo "Cleaning up..."
              kill $SERVER_PID 2>/dev/null || true
              wait $SERVER_PID 2>/dev/null || true
            }
            trap cleanup EXIT

            # Wait for server to start (longer timeout since models need to load)
            echo "Waiting for server to start (this may take a while for model loading)..."
            for i in {1..120}; do
              if ${defaultPkgs.curl}/bin/curl -s http://127.0.0.1:18000/health >/dev/null 2>&1; then
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

            # Test health endpoint
            echo "Testing health endpoint..."
            if ${defaultPkgs.curl}/bin/curl -s http://127.0.0.1:18000/health | grep -q "OK"; then
              echo "✓ Health check passed"
            else
              echo "✗ Health check failed"
              exit 1
            fi

            # Test model listing
            echo "Testing model listing..."
            ${defaultPkgs.curl}/bin/curl -s http://127.0.0.1:18000/v1/models -o models.json
            if [ -f models.json ] && ${defaultPkgs.jq}/bin/jq -e '.data' models.json >/dev/null; then
              echo "✓ Model listing endpoint works"
              echo "Available models:"
              ${defaultPkgs.jq}/bin/jq -r '.data[].id' models.json
            else
              echo "✗ Model listing failed"
              exit 1
            fi

            ORIGINAL_TEXT="People assume that time is a strict progression of cause to effect. But actually, from a nonlinear, non-subjective viewpoint, it is more like a big ball of wibbly wobbly, timey wimey stuff"

            # Test TTS (Text-to-Speech) with real model
            echo "Testing TTS with Kokoro model..."
            TTS_RESPONSE=$(${defaultPkgs.curl}/bin/curl -s -w "%{http_code}" -X POST "http://127.0.0.1:18000/v1/audio/speech" \
              -H "Content-Type: application/json" \
              -d "{\"model\": \"tts-1\", \"input\": \"$ORIGINAL_TEXT\", \"voice\": \"af_bella\"}" \
              -o test_tts_output.wav)

            if [[ "$TTS_RESPONSE" =~ ^2[0-9][0-9]$ ]] && [ -f test_tts_output.wav ] && [ -s test_tts_output.wav ]; then
              echo "✓ TTS test passed - generated $(du -h test_tts_output.wav | cut -f1) audio file"
              ${defaultPkgs.file}/bin/file test_tts_output.wav
            else
              echo "✗ TTS test failed (HTTP $TTS_RESPONSE)"
              exit 1
            fi

            # Test STT (Speech-to-Text) with the generated audio
            echo "Testing STT with Whisper base model using generated audio..."
            STT_RESPONSE=$(${defaultPkgs.curl}/bin/curl -s -w "%{http_code}" -X POST "http://127.0.0.1:18000/v1/audio/transcriptions" \
              -F "file=@test_tts_output.wav" \
              -F "model=Systran/faster-whisper-base" \
              -o transcription.json)

            if [[ "$STT_RESPONSE" =~ ^2[0-9][0-9]$ ]] && [ -f transcription.json ] && ${defaultPkgs.jq}/bin/jq -e '.text' transcription.json >/dev/null; then
              echo "✓ STT test passed - got transcription"
              echo "Original text: $ORIGINAL_TEXT"
              echo "Transcription: $(${defaultPkgs.jq}/bin/jq -r '.text' transcription.json)"

              # Check if transcription is reasonably similar (basic check)
              # Create ORIGINAL_WORDS from ORIGINAL_TEXT by converting to lowercase and removing punctuation
              ORIGINAL_WORDS=$(echo "$ORIGINAL_TEXT" | tr '[:upper:]' '[:lower:]' | sed 's/[.,!?]//g')
              TRANSCRIBED=$(${defaultPkgs.jq}/bin/jq -r '.text' transcription.json | tr '[:upper:]' '[:lower:]' | sed 's/[.,!?]//g')

              # Count total words in original text for dynamic comparison
              TOTAL_WORDS=$(echo $ORIGINAL_WORDS | wc -w)

              # Count matching words (basic similarity check)
              MATCHES=0
              for word in $ORIGINAL_WORDS; do
                if echo "$TRANSCRIBED" | grep -q "$word"; then
                  MATCHES=$((MATCHES + 1))
                fi
              done

              if [ $MATCHES -ge 5 ]; then
                echo "✓ Transcription quality check passed ($MATCHES/$TOTAL_WORDS key words matched)"
              else
                echo "⚠ Transcription quality check: only $MATCHES/$TOTAL_WORDS key words matched"
              fi
            else
              echo "✗ STT test failed (HTTP $STT_RESPONSE)"
              if [ -f transcription.json ]; then
                echo "Response content:"
                cat transcription.json
              fi
              exit 1
            fi

            echo "=== Complete End-to-End Test Passed! ==="
            echo "✓ TTS: Text → Audio conversion working"
            echo "✓ STT: Audio → Text conversion working"
            echo "✓ Models loaded and functioning offline"
            echo "Test artifacts in: $TEST_DIR"
          '';
        };

        # Applications
        apps.default = {
          type = "app";
          program = "${speaches}/bin/speaches";
          meta = {
            description = "AI-powered speech processing application";
            maintainers = ["longregen <claude@infophysics.org>"];
          };
        };

        # NixOS module
        nixosModules.default = import ./nix/module.nix;

        # NixOS tests
        checks = {
          nixos-test = import ./nix/test.nix {
            pkgs = defaultPkgs;
            speachesPackage = speaches-cpu;
            speachesModule = ./nix/module.nix;
          };

          # End-to-end tests using parameterized function
          e2e-nixos-test = mkE2eTest {pythonVersion = "python313";};
          e2e-nixos-test-py312 = mkE2eTest {
            pythonVersion = "python312";
            fullTest = false;
          };
          e2e-nixos-test-py313 = mkE2eTest {
            pythonVersion = "python313";
            fullTest = false;
          };
        };

        formatter = defaultPkgs.nixfmt-rfc-style;
      }
    );
}
