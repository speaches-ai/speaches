# Standalone e2e test (no VM required)
{
  pkgs,
  speachesPackage,
  modelCache,
}:
pkgs.writeShellScriptBin "speaches-e2e-test" ''
  set -euo pipefail

  echo "=== Speaches E2E Test ==="

  export HF_HUB_CACHE="${modelCache}/hub"
  export HF_HUB_OFFLINE=1

  TEST_DIR=$(mktemp -d)
  cd "$TEST_DIR"
  echo "Test directory: $TEST_DIR"

  echo "Starting server..."
  ${speachesPackage}/bin/speaches --host 127.0.0.1 --port 18000 &
  SERVER_PID=$!

  cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
  }
  trap cleanup EXIT

  echo "Waiting for server..."
  for i in $(seq 1 120); do
    if ${pkgs.curl}/bin/curl -s http://127.0.0.1:18000/health >/dev/null 2>&1; then
      echo "Server ready!"
      break
    fi
    [ $i -eq 120 ] && { echo "Server timeout"; exit 1; }
    [ $((i % 10)) -eq 0 ] && echo "Waiting... ($i/120)"
    sleep 1
  done

  echo "Testing health..."
  ${pkgs.curl}/bin/curl -f http://127.0.0.1:18000/health

  echo "Testing models endpoint..."
  ${pkgs.curl}/bin/curl -s http://127.0.0.1:18000/v1/models | ${pkgs.jq}/bin/jq -e '.data'

  ORIGINAL_TEXT="Hello world, this is a test of text to speech."

  echo "Testing TTS..."
  ${pkgs.curl}/bin/curl -s -X POST "http://127.0.0.1:18000/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"tts-1\", \"input\": \"$ORIGINAL_TEXT\", \"voice\": \"af_bella\"}" \
    -o test.wav

  [ -s test.wav ] || { echo "TTS failed"; exit 1; }
  echo "TTS OK: $(${pkgs.coreutils}/bin/du -h test.wav | cut -f1)"

  echo "Testing STT..."
  TRANSCRIPTION=$(${pkgs.curl}/bin/curl -s -X POST "http://127.0.0.1:18000/v1/audio/transcriptions" \
    -F "file=@test.wav" \
    -F "model=Systran/faster-whisper-base")

  echo "Transcription: $TRANSCRIPTION"
  echo "$TRANSCRIPTION" | ${pkgs.jq}/bin/jq -e '.text' || { echo "STT failed"; exit 1; }

  echo "=== E2E Test Passed ==="
''
