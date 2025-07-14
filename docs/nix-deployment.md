# Nix Deployment Guide for Speaches

This guide covers how to deploy Speaches using Nix instead of Docker.

## Prerequisites

- Nix with flakes enabled
- For GPU support: Linux with NVIDIA drivers

## Quick Start

### Running Directly

```bash
# Run with GPU support (Linux only)
nix run github:speaches-ai/speaches

# Run CPU-only version
nix run github:speaches-ai/speaches#speaches-cpu
```

### Development Environment

```bash
# Enter development shell
nix develop

# The shell includes all dependencies and tools
```

## Installation Methods

### 1. NixOS System Service

Add to your NixOS configuration:

```nix
{
  inputs.speaches.url = "github:speaches-ai/speaches";
  
  outputs = { self, nixpkgs, speaches, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        speaches.nixosModules.default
        {
          services.speaches = {
            enable = true;
            port = 8000;
            host = "0.0.0.0";
            enableCuda = true;  # Set to false for CPU-only
            
            # Optional: provide environment file with secrets
            environmentFile = /etc/speaches/secrets.env;
          };
        }
      ];
    };
  };
}
```

### 2. Direct Package Installation

```nix
{
  environment.systemPackages = [
    speaches.packages.${system}.speaches
  ];
}
```

## Configuration Options

### NixOS Module Options

- `enable`: Enable the Speaches service
- `package`: The Speaches package to use (defaults to GPU-enabled)
- `port`: Port to listen on (default: 8000)
- `host`: Host to bind to (default: "0.0.0.0")
- `enableCuda`: Enable CUDA support (default: true)
- `environmentFile`: Path to environment file with secrets
- `dataDir`: Directory for data and cache (default: "/var/lib/speaches")

### Environment Variables

Create an environment file (e.g., `/etc/speaches/secrets.env`):

```bash
# All these are optional
LOG_LEVEL=INFO
HF_TOKEN=hf_...
```

## Testing

### Run NixOS VM Tests

```bash
# Run basic service test
nix build .#checks.x86_64-linux.nixos-test --print-build-logs

# Run comprehensive integration test
nix build .#checks.x86_64-linux.nixos-integration-test --print-build-logs
```

### Manual Testing

```bash
# Check service status
systemctl status speaches

# View logs
journalctl -u speaches -f

# Test API
curl http://localhost:8000/v1/models
```

## Building

### Build Package

```bash
# Build GPU version
nix build .#speaches

# Build CPU-only version
nix build .#speaches-cpu

# Result will be in ./result/
```

### Build and Run

```bash
# Build and run immediately
nix run .#speaches

# With custom arguments
nix run .#speaches -- --port 8080
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

1. Ensure NVIDIA drivers are installed
2. Check GPU is accessible: `nvidia-smi`
3. Use CPU-only version: `enableCuda = false`
  ```nix
  services.speaches = {
    enable = true;
    enableCuda = false;
  };
  ```

### Permission Errors

The service runs as the `speaches` user. Ensure:
- Data directory has correct permissions
- Environment file is readable by the service

### Model Download Issues

Models are downloaded to `$dataDir/huggingface/hub`. If downloads fail:
- Use `nix-hug` to fetch the correct values for hashes
- Check internet connectivity
- Ensure sufficient disk space
- Check HuggingFace token if using private models

## Advanced

### Language Support Notes

The Kokoro TTS model uses the [misaki](https://github.com/hexgrad/misaki) project for text processing, which has different dependencies based on the target language. The base Speaches package has been tested with English, but other languages may require additional dependencies.

#### Language-Specific Dependencies (from misaki)

- **English** (tested): `num2words`, `spacy`, `spacy-curated-transformers`, `phonemizer-fork`, `espeakng-loader`, `torch`, `transformers`
- **Japanese**: `fugashi`, `jaconv`, `mojimoji`, `unidic`, `pyopenjtalk`
- **Korean**: `jamo`, `nltk`
- **Chinese**: `jieba`, `ordered-set`, `pypinyin`, `cn2an`, `pypinyin-dict`
- **Vietnamese**: `num2words`, `spacy`, `spacy-curated-transformers`, `underthesea`
- **Hebrew**: `mishkal-hebrew>=0.3.2`

### Adding Language Support

To extend Speaches with support for additional languages, you'll need to create a custom package with the required dependencies:

```nix
{ pkgs, ... }:

let
  # Example: Adding Japanese support
  speachesWithJapanese = pkgs.speaches.overrideAttrs (oldAttrs: {
    propagatedBuildInputs = oldAttrs.propagatedBuildInputs ++ (with pkgs.python3Packages; [
      fugashi
      jaconv
      mojimoji
      unidic
      pyopenjtalk
    ]);
  });
in
{
  services.speaches = {
    enable = true;
    package = speachesWithJapanese;
  };
}
```

Note: Some of these dependencies may not be available in nixpkgs and might need to be packaged separately. The English dependencies are included by default in the Speaches package.

